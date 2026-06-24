"""
app.py — Gradio web UI for Prof AI.
Run: python app.py  →  http://127.0.0.1:7860
"""

import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import gradio as gr
from scipy.signal import resample as _scipy_resample

from batch_transcribe import transcribe_file
from gaps import gaps_stream, resolve_session_id
from ingest import DOCS_DIR, ingest_paths
from live_gap import LiveGapWorker
from live_transcribe import (
    get_current_session, start_live_session, stop_live_session,
    pause_recording, resume_recording,
    push_audio, push_question_audio,
    QA_BUFFER_SECONDS, QA_TAIL_SECONDS,
)
from manifest import load_manifest, save_manifest
from module_store import (
    load_modules, save_modules, get_module,
    create_module, rename_module, delete_module,
    add_docs_to_module, remove_docs_from_module,
    set_module_docs, reorder_module_docs,
    expand_selection, module_dd_choices, unified_source_choices,
    load_classes, get_class, create_class, rename_class, delete_class,
    assign_module_to_class, class_tree, class_dd_choices,
)
from qa import answer_question, answer_conversation_turn
from query import query_stream, search_literature
from session import Mode
from tts import PiperTTS
from ui_theme import THEME, CSS, FORCE_DARK_JS, HEADER_HTML

SESSIONS_MANIFEST = Path("sessions.json")
DOC_MANIFEST = Path("manifest.json")
LECTURES_DIR = Path("lectures")
QA_AUDIO_DIR = Path("transcripts") / "qa_audio"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}
DOC_EXTS = {".pdf", ".txt", ".docx", ".pptx"}

# ---------------------------------------------------------------------------
# Live gap analysis shared state
# ---------------------------------------------------------------------------

_live_gap_result: str = ""
_live_gap_updated: datetime | None = None
_live_gap_lock = threading.Lock()
_live_gap_worker: LiveGapWorker | None = None

# ---------------------------------------------------------------------------
# Mode status — updated by background thread, polled by timer
# ---------------------------------------------------------------------------

_mode_status: str = "Stopped"
_mode_status_lock = threading.Lock()


def _set_status(text: str) -> None:
    global _mode_status
    with _mode_status_lock:
        _mode_status = text


def _get_status() -> str:
    with _mode_status_lock:
        return _mode_status


# ---------------------------------------------------------------------------
# TTS singleton — lazy-loaded; silently disabled if model not found
# ---------------------------------------------------------------------------

_tts: PiperTTS | None = None
_tts_available: bool = True


def _get_tts() -> PiperTTS | None:
    global _tts, _tts_available
    if not _tts_available:
        return None
    if _tts is None:
        try:
            _tts = PiperTTS()
        except Exception:
            print("[TTS] Failed to load — voice disabled:", flush=True)
            traceback.print_exc()
            _tts_available = False
            return None
    return _tts


# ---------------------------------------------------------------------------
# Q&A audio — written by background thread, polled by UI timer
# ---------------------------------------------------------------------------

_qa_audio_path: str | None = None
_qa_audio_lock = threading.Lock()

_conv_display: str = ""
_conv_display_lock = threading.Lock()
_conv_pending_answer: str = ""
_conv_pending_lock = threading.Lock()


def _init_qa_audio_dir() -> None:
    QA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 3600
    for f in QA_AUDIO_DIR.glob("*.wav"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _save_qa_audio(wav_bytes: bytes) -> str:
    QA_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = QA_AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
    path.write_bytes(wav_bytes)
    return str(path)


def _set_qa_audio(path: str | None) -> None:
    global _qa_audio_path
    with _qa_audio_lock:
        _qa_audio_path = path


def _get_and_clear_qa_audio() -> str | None:
    global _qa_audio_path
    with _qa_audio_lock:
        path = _qa_audio_path
        _qa_audio_path = None
    return path


def _clear_conv_state() -> None:
    global _conv_display, _conv_pending_answer
    with _conv_display_lock:
        _conv_display = ""
    with _conv_pending_lock:
        _conv_pending_answer = ""


def _set_conv_display(session) -> None:
    global _conv_display
    lines = []
    for entry in getattr(session, "conversation_history", []):
        lines.append(f"Q: {entry['question']}")
        lines.append(f"A: {entry['answer']}")
        lines.append("")
    with _conv_display_lock:
        _conv_display = "\n".join(lines).strip()


def _resume_after_playback(session) -> None:
    """Called by threading.Timer after estimated audio duration elapses."""
    if session and session.current_mode() == Mode.PROCESSING:
        resume_recording()
        session.set_mode(Mode.LECTURE)
        _set_status("● Recording lecture")
        print("[Q&A] mic resumed after estimated playback duration", flush=True)
    else:
        print("[Q&A] playback timer fired — mode already changed, skipping", flush=True)


def _resume_after_conv_turn(session) -> None:
    """Called by threading.Timer after conversation turn TTS finishes."""
    if session and session.current_mode() == Mode.PROCESSING:
        session.set_mode(Mode.LECTURE)
        _set_status("● Conversation — ready")
        print("[CONV] ready for next turn", flush=True)
    else:
        print("[CONV] resume timer fired — mode already changed, skipping", flush=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def scope_choices() -> list[str]:
    sessions = sorted(load_manifest(SESSIONS_MANIFEST).keys(), reverse=True)
    return ["All material", "Latest session"] + sessions


def scope_to_session_id(scope: str) -> str | None:
    if scope == "All material":
        return None
    if scope == "Latest session":
        sessions = load_manifest(SESSIONS_MANIFEST)
        return max(sessions.keys()) if sessions else None
    return scope


def doc_dropdown_choices() -> list[str]:
    return sorted({Path(k).name for k in load_manifest(DOC_MANIFEST) if Path(k).suffix.lower() in DOC_EXTS})


import json as _json


def _render_drag_panel() -> str:
    """Generate the four-pane drag-drop module manager HTML.

    Pane 1 — Classes: click to select, SortableJS drop target for module reassignment.
    Pane 2 — Modules in selected class: click to select, drag source for class reassignment.
    Pane 3 — Documents in selected module: reorder + accept library drops.
    Pane 4 — Document Library: clone-pull source with filter input.

    Script bootstrap: onerror on a tiny img — plain <script> tags in innerHTML don't execute.
    Quoting: &#39; for single-quotes inside onclick attribute values (not \' — Python eats
    the backslash and the bare ' breaks JS string literals, causing silent SyntaxError).
    """
    tree = class_tree()
    library = sorted({val for label, val in unified_source_choices() if label.startswith("[Doc]")})

    doc_count: dict[str, int] = {}
    for group in tree:
        for m in group["modules"]:
            for fn in m.get("documents", []):
                doc_count[fn] = doc_count.get(fn, 0) + 1
    shared = [fn for fn, cnt in doc_count.items() if cnt > 1]

    tree_data = [
        {
            "class_id": g["class_id"] or "",
            "name": g["name"],
            "modules": [
                {"id": m["id"], "name": m["name"], "documents": list(m.get("documents", []))}
                for m in g["modules"]
            ],
        }
        for g in tree
    ]
    payload = _json.dumps({"tree": tree_data, "library": library, "shared": shared},
                          ensure_ascii=False)

    return f"""<style>
.pma-ghost {{ opacity:.4; background:var(--pa-accent-bg) !important; }}
.pma-item {{ cursor:grab; }}
.pma-item:active {{ cursor:grabbing; }}
.pma-mod-dragging .pma-class-drop {{ outline:2px dashed var(--pa-accent-border) !important; background:var(--pa-accent-bg-soft) !important; }}
.pma-lib-dragging #pma-doc-list {{ outline:2px dashed var(--pa-accent-border) !important; background:var(--pa-accent-bg-soft) !important; }}
.pma-hidden-bridge {{ position:absolute !important; left:-9999px !important; width:1px !important; height:1px !important; overflow:hidden !important; }}
</style>
<div id="profai-mod-manager" style="display:flex;gap:0;height:460px;border:1px solid var(--pa-border);border-radius:8px;overflow:hidden;font-family:system-ui,sans-serif;font-size:13px;color:var(--pa-text);">

  <!-- Pane 1: Classes -->
  <div style="flex:1 1 0;min-width:0;border-right:1px solid var(--pa-border);display:flex;flex-direction:column;background:var(--pa-bg-2);">
    <div style="padding:8px 10px;font-weight:600;background:var(--pa-bg-3);border-bottom:1px solid var(--pa-border);font-size:11px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Classes</div>
    <div style="padding:3px 8px 4px;font-size:10px;color:var(--pa-muted-2);border-bottom:1px solid var(--pa-border-faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Drag a module here to assign it to this class.</div>
    <div id="pma-class-list" style="flex:1;overflow-y:auto;"><div style="padding:12px;color:var(--pa-muted-2);font-style:italic;">Loading…</div></div>
  </div>

  <!-- Pane 2: Modules in selected class -->
  <div style="flex:1 1 0;min-width:0;border-right:1px solid var(--pa-border);display:flex;flex-direction:column;background:var(--pa-bg-1);">
    <div id="pma-class-label" style="padding:8px 10px;font-weight:600;background:var(--pa-bg-3);border-bottom:1px solid var(--pa-border);font-size:11px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Modules</div>
    <div id="pma-mod-list" style="flex:1;overflow-y:auto;"><div style="padding:12px;color:var(--pa-muted-2);font-style:italic;">Loading…</div></div>
  </div>

  <!-- Pane 3: Documents in selected module -->
  <div style="flex:1 1 0;min-width:0;border-right:1px solid var(--pa-border);display:flex;flex-direction:column;background:var(--pa-bg-1);">
    <div style="padding:8px 10px;font-weight:600;background:var(--pa-bg-3);border-bottom:1px solid var(--pa-border);font-size:11px;letter-spacing:.06em;text-transform:uppercase;display:flex;align-items:center;gap:6px;overflow:hidden;">
      <span id="pma-mod-label" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">Documents</span>
      <span id="pma-doc-count" style="font-weight:400;color:var(--pa-muted-2);white-space:nowrap;flex-shrink:0;"></span>
    </div>
    <div style="padding:3px 8px 4px;font-size:10px;color:var(--pa-muted-2);border-bottom:1px solid var(--pa-border-faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Drag documents here from the library &middot; drag to reorder &middot; &times; to remove.</div>
    <div id="pma-doc-list" style="flex:1;overflow-y:auto;padding:4px 0;min-height:40px;transition:background .15s;"><div style="padding:16px;color:var(--pa-muted-2);font-style:italic;">Loading…</div></div>
  </div>

  <!-- Pane 4: Document Library -->
  <div style="flex:1 1 0;min-width:0;display:flex;flex-direction:column;background:var(--pa-bg-2);">
    <div style="padding:8px 10px;font-weight:600;background:var(--pa-bg-3);border-bottom:1px solid var(--pa-border);font-size:11px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Document Library</div>
    <div style="padding:5px 8px;border-bottom:1px solid var(--pa-border);background:var(--pa-bg-3);">
      <input id="pma-lib-search" type="text" placeholder="Filter…"
        style="width:100%;box-sizing:border-box;padding:4px 7px;font-size:12px;border:1px solid var(--pa-border-2);border-radius:4px;outline:none;background:var(--pa-bg-1);color:var(--pa-text);"
        oninput="window.__pmaFilter(this.value)">
    </div>
    <div id="pma-lib-list" style="flex:1;overflow-y:auto;padding:4px 0;"><div style="padding:12px;color:var(--pa-muted-2);font-style:italic;">Loading…</div></div>
  </div>

</div>
<script type="application/json" id="pma-data">{payload}</script>
<script type="text/x-pma" id="pma-init">
(function(DATA){{
  const TREE    = DATA.tree;
  const LIB     = DATA.library;
  const SHARED  = new Set(DATA.shared);

  const MODS_BY_ID = {{}};
  TREE.forEach(function(g){{ g.modules.forEach(function(m){{ MODS_BY_ID[m.id] = m; }}); }});

  // Selection state — class_id encoded as "" for Unassigned.
  // The panel is fully re-rendered after every mutation, which used to reset
  // selection to the first class/module. Selection is now persisted in
  // localStorage and restored on each rebuild so the user keeps their place.
  var SEL_CLASS = '';
  var SEL_MOD   = null;
  var DOCS      = [];

  function saveSel(){{
    try {{
      localStorage.setItem('pma-sel-class', SEL_CLASS);
      localStorage.setItem('pma-sel-mod', SEL_MOD || '');
    }} catch(e) {{}}
  }}

  // Restore previous selection; fall back to first class / first module.
  (function(){{
    var sc = null, sm = null;
    try {{
      sc = localStorage.getItem('pma-sel-class');
      sm = localStorage.getItem('pma-sel-mod');
    }} catch(e) {{}}
    if (sm === '') sm = null;

    // Prefer the stored module: find whichever class currently contains it
    // (it may have been dragged to a different class since last render).
    var g = null;
    if (sm) {{
      for (var i = 0; i < TREE.length; i++) {{
        if (TREE[i].modules.some(function(m){{ return m.id === sm; }})) {{ g = TREE[i]; break; }}
      }}
      if (!g) sm = null;  // stored module no longer exists
    }}
    if (!g && sc !== null) {{
      g = TREE.find(function(grp){{ return (grp.class_id || '') === sc; }}) || null;
    }}
    if (!g) g = TREE.length > 0 ? TREE[0] : null;

    SEL_CLASS = g ? (g.class_id || '') : '';
    if (!sm && g && g.modules.length > 0) sm = g.modules[0].id;
    SEL_MOD = sm || null;
    DOCS = (SEL_MOD && MODS_BY_ID[SEL_MOD]) ? MODS_BY_ID[SEL_MOD].documents.slice() : [];
  }})();

  function esc(s){{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}
  function escA(s){{ return String(s).replace(/'/g,'&#39;').replace(/"/g,'&quot;'); }}

  function bridgeDrag(){{ return document.querySelector('#drag_state textarea'); }}
  function bridgeClass(){{ return document.querySelector('#drag_state textarea'); }}

  function push(){{
    var el = bridgeDrag();
    if (!el || !SEL_MOD) return;
    el.value = JSON.stringify({{module_id: SEL_MOD, ordered_filenames: DOCS}});
    var t = document.querySelector('#drag_trigger');
    if (t) t.click();
  }}

  function pushClass(module_id, class_id){{
    var el = bridgeClass();
    if (!el) return;
    el.value = JSON.stringify({{module_id: module_id, class_id: class_id || null}});
    var t = document.querySelector('#drag_trigger');
    if (t) t.click();
  }}

  function getClassGroup(cid){{
    return TREE.find(function(g){{ return (g.class_id || '') === cid; }}) || null;
  }}

  // ── Pane 1: Classes ───────────────────────────────────────────────────────
  function renderClasses(){{
    var el = document.getElementById('pma-class-list');
    if (!el) return;
    if (!TREE.length) {{
      el.innerHTML = '<div style="padding:12px;color:var(--pa-muted-2);font-style:italic;">No classes yet</div>';
      return;
    }}
    var html = '';
    TREE.forEach(function(g){{
      var cid = g.class_id || '';
      var act = cid === SEL_CLASS;
      var cnt = g.modules.length;
      html += '<div class="pma-class-drop" data-class-id="'+escA(cid)+'"'
            + ' onclick="window.__pmaSelClass(&#39;'+escA(cid)+'&#39;)"'
            + ' style="padding:8px 10px;cursor:pointer;border-bottom:1px solid var(--pa-bg-3);'
            + 'background:'+(act?'var(--pa-accent-bg)':'var(--pa-bg-1)')+';'
            + 'border-left:3px solid '+(act?'var(--pa-accent)':'transparent')+';'
            + 'display:flex;align-items:center;gap:6px;">'
            + '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
            + 'font-weight:'+(act?600:400)+';font-size:12px;color:'+(act?'var(--pa-link)':'var(--pa-text-2)')+';"'
            + ' title="'+escA(g.name)+'">'+esc(g.name)+'</span>'
            + '<span style="font-size:10px;color:var(--pa-muted-2);white-space:nowrap;flex-shrink:0;">'
            + cnt+' mod'+(cnt!==1?'s':'')+'</span>'
            + '</div>';
    }});
    el.innerHTML = html;
    initClassDropTargets();
  }}

  // ── Pane 2: Modules in selected class ─────────────────────────────────────
  function renderModules(){{
    var el  = document.getElementById('pma-mod-list');
    var lbl = document.getElementById('pma-class-label');
    if (!el) return;
    var g = getClassGroup(SEL_CLASS);
    if (lbl) lbl.textContent = g ? g.name : '—';
    var mods = g ? g.modules : [];
    if (!mods.length) {{
      el.innerHTML = '<div style="padding:12px;color:var(--pa-muted-2);font-style:italic;">No modules in this class yet.</div>';
      if (el._sort) {{ el._sort.destroy(); el._sort = null; }}
      return;
    }}
    var html = '';
    mods.forEach(function(m){{
      var act = m.id === SEL_MOD;
      var cnt = act ? DOCS.length : m.documents.length;
      html += '<div data-mod-id="'+escA(m.id)+'" class="pma-item pma-mod-item"'
            + ' onclick="window.__pmaSel(&#39;'+escA(m.id)+'&#39;)"'
            + ' style="padding:8px 10px;cursor:grab;border-bottom:1px solid var(--pa-bg-3);'
            + 'background:'+(act?'var(--pa-accent-bg)':'var(--pa-bg-1)')+';'
            + 'border-left:3px solid '+(act?'var(--pa-accent)':'transparent')+';'
            + 'color:'+(act?'var(--pa-link)':'var(--pa-text-2)')+';">'
            + '<div style="font-weight:'+(act?600:400)+';white-space:nowrap;overflow:hidden;'
            + 'text-overflow:ellipsis;font-size:12px;" title="'+escA(m.name)+'">'+esc(m.name)+'</div>'
            + '<div style="font-size:10px;color:var(--pa-muted-2);">'+cnt+' doc'+(cnt!==1?'s':'')+'</div>'
            + '</div>';
    }});
    el.innerHTML = html;
    initModuleDragSource();
  }}

  // ── Pane 3: Documents in selected module ──────────────────────────────────
  function renderDocs(){{
    var el  = document.getElementById('pma-doc-list');
    var lbl = document.getElementById('pma-mod-label');
    var cnt = document.getElementById('pma-doc-count');
    if (!el) return;
    var mod = MODS_BY_ID[SEL_MOD];
    if (lbl) lbl.textContent = mod ? mod.name : '— select a module —';
    if (cnt) cnt.textContent = SEL_MOD ? DOCS.length+' doc'+(DOCS.length!==1?'s':'') : '';
    if (!SEL_MOD) {{
      el.innerHTML = '<div style="padding:16px;color:var(--pa-muted-2);font-style:italic;">Select a module ←</div>';
      return;
    }}
    el.innerHTML = DOCS.length ? DOCS.map(function(fn,i){{
      return '<div data-fn="'+escA(fn)+'" class="pma-item"'
           + ' style="display:flex;align-items:center;gap:6px;padding:6px 8px;border-bottom:1px solid var(--pa-bg-3);background:var(--pa-bg-1);">'
           + '<span style="color:var(--pa-muted-2);min-width:18px;font-size:11px;text-align:right;user-select:none;">'+(i+1)+'</span>'
           + '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;" title="'+escA(fn)+'">'+(SHARED.has(fn)?'&#128279;&nbsp;':'')+esc(fn)+'</span>'
           + '<button onclick="window.__pmaRm(&#39;'+escA(fn)+'&#39;)"'
           + ' style="border:none;background:none;cursor:pointer;color:var(--pa-muted-2);padding:2px 5px;font-size:15px;line-height:1;border-radius:3px;"'
           + ' title="Remove">&times;</button>'
           + '</div>';
    }}).join('') : '<div style="padding:16px;color:var(--pa-muted-2);font-style:italic;">No documents in this module yet. Drag from the library →</div>';
    initDocSort();
  }}

  // ── Pane 4: Document Library ──────────────────────────────────────────────
  function renderLib(){{
    var el = document.getElementById('pma-lib-list');
    if (!el) return;
    var cur = new Set(DOCS);
    el.innerHTML = LIB.length ? LIB.map(function(fn){{
      var here = cur.has(fn);
      return '<div data-fn="'+escA(fn)+'" '+(here?'':'class="pma-item"')
           + ' style="display:flex;align-items:center;gap:6px;padding:7px 10px;border-bottom:1px solid var(--pa-bg-3);'
           + 'opacity:'+(here?.45:1)+';background:'+(here?'var(--pa-bg-2)':'var(--pa-bg-1)')+';cursor:'+(here?'default':'grab')+';"'
           + ' title="'+(here?'Already in module':escA(fn))+'">'
           + '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;">'+(SHARED.has(fn)?'&#128279;&nbsp;':'')+esc(fn)+'</span>'
           + (here?'<span style="color:var(--pa-muted-2);font-size:11px;">&#10003;</span>':'')
           + '</div>';
    }}).join('') : '<div style="padding:16px;color:var(--pa-muted-2);font-style:italic;">No documents ingested yet</div>';
    initLibSort();
    var sq = document.getElementById('pma-lib-search');
    if (sq && sq.value) window.__pmaFilter(sq.value);
  }}

  // ── SortableJS init ───────────────────────────────────────────────────────

  // Pane 1: class rows accept module drops via native HTML5 DnD.
  // SortableJS onAdd is unreliable when the drop target contains only inline
  // children (spans) rather than same-type list items — it silently mis-fires.
  // Native dragover/drop events on the row divs are straightforward and do not
  // conflict with SortableJS managing the source list in pane 2.
  function initClassDropTargets(){{
    document.querySelectorAll('.pma-class-drop').forEach(function(row){{
      row.addEventListener('dragover', function(e){{ e.preventDefault(); }});
      row.addEventListener('dragenter', function(){{ row.style.outline='2px dashed var(--pa-accent)'; }});
      row.addEventListener('dragleave', function(){{ row.style.outline=''; }});
      row.addEventListener('drop', function(e){{
        e.preventDefault();
        row.style.outline='';
        var modId = e.dataTransfer.getData('pma-mod-id');
        if (!modId) return;
        pushClass(modId, row.getAttribute('data-class-id') || null);
      }});
    }});
  }}

  // Pane 2: module list. SortableJS drives the visual drag; a native dragstart
  // listener tags dataTransfer so class rows (native DnD targets) can read it.
  function initModuleDragSource(){{
    var el = document.getElementById('pma-mod-list');
    if (!el || !window.Sortable) return;
    if (el._sort) el._sort.destroy();
    var mgr = document.getElementById('profai-mod-manager');
    el._sort = Sortable.create(el, {{
      animation: 150, ghostClass: 'pma-ghost',
      group: {{name:'pmaclasses', pull:true, put:false}},
      sort: false,
      onStart: function(){{ if (mgr) mgr.classList.add('pma-mod-dragging'); }},
      onEnd: function(){{ if (mgr) mgr.classList.remove('pma-mod-dragging'); }},
    }});
    el.addEventListener('dragstart', function(e){{
      var item = e.target.closest ? e.target.closest('[data-mod-id]') : e.target;
      if (item && item.getAttribute('data-mod-id')) {{
        e.dataTransfer.setData('pma-mod-id', item.getAttribute('data-mod-id'));
      }}
    }});
  }}

  // Pane 3: doc list accepts drops from library and supports intra-list reorder
  function initDocSort(){{
    var el = document.getElementById('pma-doc-list');
    if (!el || !window.Sortable) return;
    if (el._sort) el._sort.destroy();
    el._sort = Sortable.create(el, {{
      animation:150, ghostClass:'pma-ghost', group:'pmadocs',
      filter:'.pma-drop-placeholder',
      onEnd: function(){{ syncDocs(el); renderLib(); push(); }},
      onAdd: function(){{ syncDocs(el); renderLib(); push(); }},
    }});
  }}

  // Pane 4: library is a clone-pull source only
  function initLibSort(){{
    var el = document.getElementById('pma-lib-list');
    if (!el || !window.Sortable) return;
    if (el._sort) el._sort.destroy();
    var mgr = document.getElementById('profai-mod-manager');
    el._sort = Sortable.create(el, {{
      animation:150, ghostClass:'pma-ghost',
      group:{{name:'pmadocs', pull:'clone', put:false}},
      filter:'[style*="cursor: default"],[style*="cursor:default"]',
      preventOnFilter:true,
      sort:false,
      onStart: function(){{ if (mgr) mgr.classList.add('pma-lib-dragging'); }},
      onEnd: function(){{ if (mgr) mgr.classList.remove('pma-lib-dragging'); }},
    }});
  }}

  function syncDocs(el){{
    DOCS = Array.from(el.querySelectorAll('[data-fn]')).map(function(c){{ return c.getAttribute('data-fn'); }}).filter(Boolean);
    var cnt = document.getElementById('pma-doc-count');
    if (cnt) cnt.textContent = DOCS.length+' doc'+(DOCS.length!==1?'s':'');
    el.querySelectorAll('[data-fn]').forEach(function(c,i){{
      var num = c.querySelector('span');
      if (num) num.textContent = i+1;
    }});
    var mod = MODS_BY_ID[SEL_MOD];
    if (mod) mod.documents = DOCS.slice();
  }}

  // ── Global handlers ───────────────────────────────────────────────────────

  window.__pmaSelClass = function(cid){{
    SEL_CLASS = cid;
    var g = getClassGroup(cid);
    if (g && g.modules.length > 0) {{
      SEL_MOD = g.modules[0].id;
      DOCS    = g.modules[0].documents.slice();
    }} else {{
      SEL_MOD = null;
      DOCS    = [];
    }}
    saveSel();
    renderClasses(); renderModules(); renderDocs(); renderLib();
  }};

  window.__pmaSel = function(id){{
    var mod = MODS_BY_ID[SEL_MOD];
    if (mod) mod.documents = DOCS.slice();
    SEL_MOD = id;
    var next = MODS_BY_ID[id];
    DOCS = next ? next.documents.slice() : [];
    saveSel();
    renderModules(); renderDocs(); renderLib();
  }};

  window.__pmaRm = function(fn){{
    DOCS = DOCS.filter(function(f){{ return f !== fn; }});
    renderDocs(); renderLib(); push();
  }};

  window.__pmaFilter = function(q){{
    q = q.trim().toLowerCase();
    var el = document.getElementById('pma-lib-list');
    if (!el) return;
    el.querySelectorAll('[data-fn]').forEach(function(item){{
      var fn = (item.getAttribute('data-fn')||'').toLowerCase();
      item.style.display = (!q || fn.includes(q)) ? '' : 'none';
    }});
  }};

  // ── Init ──────────────────────────────────────────────────────────────────
  renderClasses(); renderModules(); renderDocs(); renderLib();

  if (!window.Sortable) {{
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js';
    s.onload = function(){{ initDocSort(); initLibSort(); initModuleDragSource(); }};
    s.onerror = function(){{ console.warn('[PMA] SortableJS CDN unavailable — drag reorder disabled'); }};
    document.head.appendChild(s);
  }}
}})(JSON.parse(document.getElementById('pma-data').textContent));
</script>
<img src="x" style="position:absolute;width:0;height:0;opacity:0;pointer-events:none"
     onerror="(function(){{try{{(new Function(document.getElementById('pma-init').textContent))()}}catch(e){{console.error('[PMA]',e)}}}})()">
"""


def _module_refresh():
    """Single source of dropdown + drag-panel updates after any module change.
    Returns (module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html).
    Add a new output here ONCE and every handler inherits it."""
    mod_choices = module_dd_choices()
    unified = unified_source_choices()
    return (
        gr.update(choices=mod_choices),
        gr.update(choices=unified),
        gr.update(choices=unified),
        _render_drag_panel(),
    )


# ---------------------------------------------------------------------------
# Modules tab event handlers
# ---------------------------------------------------------------------------

def create_module_fn(name: str):
    if not name.strip():
        return gr.update(), gr.update(), gr.update(), gr.update(), "", "Enter a module name."
    mod = create_module(name.strip())
    return (*_module_refresh(), "", f"Created '{mod['name']}'.")


def rename_module_fn(mod_id: str | None, new_name: str):
    if not mod_id:
        return gr.update(), gr.update(), gr.update(), gr.update(), "No module selected."
    if not new_name.strip():
        return gr.update(), gr.update(), gr.update(), gr.update(), "Enter a new name."
    rename_module(mod_id, new_name.strip())
    return (*_module_refresh(), f"Renamed to '{new_name.strip()}'.")


def delete_module_stage1(mod_id: str | None):
    if not mod_id:
        return gr.update(interactive=False), "No module selected."
    mods_by_id = {m["id"]: m for m in load_modules()}
    name = mods_by_id.get(mod_id, {}).get("name", mod_id)
    return gr.update(interactive=True), f"⚠ Click 'Confirm Delete' to remove '{name}'."


def delete_module_stage2(mod_id: str | None):
    if not mod_id:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), "No module selected."
    mods_by_id = {m["id"]: m for m in load_modules()}
    name = mods_by_id.get(mod_id, {}).get("name", mod_id)
    delete_module(mod_id)
    mod_choices = module_dd_choices()
    new_val = mod_choices[0][1] if mod_choices else None
    return (
        gr.update(choices=mod_choices, value=new_val),
        *_module_refresh()[1:],  # linked_docs_dd, doc_filter_dd, drag_panel
        gr.update(interactive=False),
        f"Deleted '{name}'.",
    )


def on_drag_change(payload_json: str):
    """Bridge handler: JS sends either {module_id, ordered_filenames} (doc reorder/add)
    or {module_id, class_id} (class assignment) through the same drag_state bridge."""
    try:
        data = _json.loads(payload_json)
        if "ordered_filenames" in data:
            set_module_docs(data["module_id"], data["ordered_filenames"])
        elif "class_id" in data:
            cid = data.get("class_id") or None
            assign_module_to_class(data["module_id"], cid)
    except Exception:
        pass
    return _module_refresh()


# ---------------------------------------------------------------------------
# Class tab event handlers
# ---------------------------------------------------------------------------

def create_class_fn(name: str):
    if not name.strip():
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "", "Enter a class name."
    cls = create_class(name.strip())
    return (*_module_refresh(), gr.update(choices=class_dd_choices()), "", f"Created class '{cls['name']}'.")


def rename_class_fn(class_id: str | None, new_name: str):
    if not class_id:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "No class selected."
    if not new_name.strip():
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "Enter a new name."
    rename_class(class_id, new_name.strip())
    return (*_module_refresh(), gr.update(choices=class_dd_choices()), f"Renamed to '{new_name.strip()}'.")


def delete_class_stage1(class_id: str | None):
    if not class_id:
        return gr.update(interactive=False), "No class selected."
    classes = {c["id"]: c for c in load_classes()}
    name = classes.get(class_id, {}).get("name", class_id)
    return gr.update(interactive=True), f"⚠ Click 'Confirm Delete Class' to remove '{name}'. Modules inside move to Unassigned."


def delete_class_stage2(class_id: str | None):
    if not class_id:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), "No class selected."
    classes = {c["id"]: c for c in load_classes()}
    name = classes.get(class_id, {}).get("name", class_id)
    delete_class(class_id)
    class_choices = class_dd_choices()
    new_class_val = class_choices[0][1] if class_choices else None
    return (
        *_module_refresh(),
        gr.update(choices=class_choices, value=new_class_val),
        gr.update(interactive=False),
        f"Deleted class '{name}'. Modules moved to Unassigned.",
    )


def refresh_modules_tab():
    mod_choices = module_dd_choices()
    return gr.update(choices=mod_choices, value=mod_choices[0][1] if mod_choices else None)


def session_label(sid: str, entry: dict) -> str:
    dt = datetime.fromisoformat(entry["start_time"])
    ts = dt.strftime("%Y-%m-%d %H:%M")
    count = entry.get("segment_count", 0)
    name = entry.get("name", "")
    if count == 0:
        return f"(empty) {ts}"
    return f"{name} — {ts} ({count} segments)" if name else f"{ts} ({count} segments)"


def gap_dropdown_choices() -> list[tuple[str, str]]:
    sessions = load_manifest(SESSIONS_MANIFEST)
    ordered = sorted(sessions.items(), key=lambda kv: kv[1].get("start_time", kv[0]), reverse=True)
    return [(session_label(sid, e), sid) for sid, e in ordered]


def refresh_gap_dropdown():
    choices = gap_dropdown_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def linked_docs_info(sid: str | None) -> tuple[str, list[str]]:
    if not sid:
        return "No session selected.", []
    entry = load_manifest(SESSIONS_MANIFEST).get(sid, {})
    linked = entry.get("linked_documents", [])
    lbl = ("Comparing against: " + ", ".join(linked)) if linked \
        else "No specific documents linked — comparing against all course material"
    return lbl, linked


def save_linked_docs_fn(sid: str | None, linked: list[str]) -> str:
    if not sid:
        return "No session selected."
    sessions = load_manifest(SESSIONS_MANIFEST)
    if sid not in sessions:
        return "Session not found."
    sessions[sid]["linked_documents"] = linked or []
    save_manifest(SESSIONS_MANIFEST, sessions)
    lbl, _ = linked_docs_info(sid)
    return lbl


def refresh_gaps_tab():
    choices = gap_dropdown_choices()
    v = choices[0][1] if choices else None
    lbl, edit = linked_docs_info(v)
    return gr.update(choices=choices, value=v), lbl, edit


# ---------------------------------------------------------------------------
# Tab 1 — Chat helpers
# ---------------------------------------------------------------------------

def _format_history(history, max_turns=6):
    """Format Gradio chat history into a plain text block for the LLM prompt.

    Handles both string content and Gradio's list-of-parts content format.
    """
    if not history:
        return ""

    recent = history[-(max_turns * 2):] if max_turns else history
    formatted = []

    for entry in recent:
        raw = entry.get("content")

        # Content can be a string OR a list of message parts. Normalize.
        if isinstance(raw, list):
            text_parts = []
            for part in raw:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts)
        elif isinstance(raw, str):
            content = raw
        else:
            content = str(raw) if raw is not None else ""

        content = content.strip()
        if not content:
            continue

        role = entry.get("role", "user").upper()
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def _format_docs_md(details: list[dict]) -> str:
    if not details:
        return "*No local documents retrieved.*"
    by_file: dict[str, list[dict]] = {}
    for d in details:
        by_file.setdefault(d["source_file"], []).append(d)
    lines = []
    for src, chunks in by_file.items():
        lines.append(f"**{src}**")
        for c in chunks:
            loc = f"`{c['location']}` " if c["location"] else ""
            lines.append(f"- {loc}{c['preview']}")
        lines.append("")
    return "\n".join(lines).strip()


_LIT_SOURCE_LABELS = {
    "PubMed (biomedical)": "pubmed",
    "Semantic Scholar (general academic)": "semantic_scholar",
    "OpenAlex (general academic, broadest coverage)": "openalex",
}


def _format_literature_html(articles: list[dict]) -> str:
    if not articles:
        return (
            "<p style='color:var(--pa-muted); font-style:italic;'>"
            "Select a literature source and ask a question to see recent papers."
            "</p>"
        )
    cards = []
    for a in articles:
        source = a.get("source", "")
        title = a.get("title", "Unknown title")
        authors = a.get("authors", [])
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

        if source == "pubmed":
            badge = "PubMed"
            venue_str = f"<strong>{a.get('journal', '')}</strong> &bull; {a.get('pub_date', '')}"
            url = a.get("url", "#")
            id_str = f"PMID {a.get('pmid', '')}"
        elif source == "semantic_scholar":
            badge = "Semantic Scholar"
            venue_str = f"<strong>{a.get('venue', '') or 'unknown venue'}</strong> &bull; {a.get('year', '')}"
            pid = a.get("id", "")
            url = f"https://www.semanticscholar.org/paper/{pid}" if pid else "#"
            id_str = pid[:8] + "…" if len(pid) > 8 else pid
        else:  # openalex
            badge = "OpenAlex"
            venue_str = f"<strong>{a.get('venue', '') or 'unknown venue'}</strong> &bull; {a.get('year', '')}"
            wid = a.get("id", "")
            doi = a.get("doi")
            url = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{wid}"
            id_str = wid

        cards.append(
            f"<div style='border:1px solid var(--pa-border-2); border-radius:8px; padding:14px;"
            f" margin:8px 0; background:var(--pa-bg-2);'>"
            f"<div style='font-size:0.75em; font-weight:600; color:var(--pa-muted);"
            f" margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em;'>{badge}</div>"
            f"<a href='{url}' target='_blank' rel='noopener noreferrer'"
            f" style='font-weight:600; font-size:0.95em; color:var(--pa-link);"
            f" text-decoration:none;'>{title}</a>"
            f"<div style='margin-top:6px; font-size:0.85em; color:var(--pa-text-3);'>"
            f"{venue_str} &bull; "
            f"<a href='{url}' target='_blank' rel='noopener noreferrer'"
            f" style='color:var(--pa-muted);'>{id_str}</a></div>"
            f"<div style='margin-top:4px; font-size:0.82em; color:var(--pa-muted);'>"
            f"{author_str}</div>"
            f"</div>"
        )
    return "".join(cards)


# ---------------------------------------------------------------------------
# Tab 1 — Chat
# ---------------------------------------------------------------------------

def chat_fn(
    message: str,
    history: list,
    scope: str,
    doc_filter: list[str],
    lit_source_labels: list[str],
    lit_max: int,
):
    session_id = scope_to_session_id(scope)
    partial, source_details, literature_articles = "", [], []

    # Expand any module IDs in the filter to their constituent filenames.
    expanded_filter = expand_selection(doc_filter) if doc_filter else []

    history_for_llm = _format_history(history, max_turns=6)
    history_for_lit = _format_history(history, max_turns=3)

    selected_sources = [_LIT_SOURCE_LABELS[s] for s in lit_source_labels if s in _LIT_SOURCE_LABELS]
    literature_results = (
        search_literature(
            message,
            sources=selected_sources,
            conversation_history=history_for_lit,
            selected_doc_titles=expanded_filter,
            max_results=int(lit_max),
        )
        if selected_sources else None
    )

    try:
        for token in query_stream(
            message, session_id,
            doc_ids=expanded_filter or None,
            literature_results=literature_results,
            conversation_history=history_for_llm,
        ):
            if isinstance(token, dict):
                source_details = token.get("source_details", [])
                literature_articles = token.get("literature", [])
            else:
                partial += token
                yield partial, "", ""
    except Exception as e:
        yield f"⚠ {e}", "", ""
        return

    yield partial, _format_docs_md(source_details), _format_literature_html(literature_articles)


# ---------------------------------------------------------------------------
# Tab 2 — Live Lecture: browser audio streaming helpers
# ---------------------------------------------------------------------------

# Noise gate + silence-gated flush.
# Chunks below _SPEECH_ENERGY_THRESHOLD are classified as noise/silence and
# are NOT accumulated into the buffer — only speech reaches Whisper.
# A per-state counter tracks consecutive noise samples so the flush still fires
# after VAD_SILENCE_MS of quiet.  Raise the threshold if background noise bleeds
# through; lower it if a soft-spoken prof gets clipped.
_VAD_SILENCE_MS = 600            # must match recorder.VAD_SILENCE_MS
_SPEECH_ENERGY_THRESHOLD = 0.02  # RMS; above = speech, below = noise/silence (lecture mode only)
_MAX_BUFFER_S = 15.0             # hard cap: flush even if silence never fires
_CONV_MAX_TURN_S = 60.0          # safety cap on conversation turn length (forgotten Done speaking)

# Question-capture flush knobs (single-turn Ask AI only; conversation is separate)
_QA_MAX_WINDOW_S = 20.0          # hard cap — flush even if silence never fires
_QA_MIN_WINDOW_S = 1.5           # don't flush on silence until at least this many seconds recorded
_AMBIENT_SPEECH_MULTIPLIER = 2.5 # adaptive speech threshold = ambient_rms * this
_AMBIENT_ALPHA = 0.05            # EMA weight for ambient calibration (lower = slower tracking)


def _resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample audio to 16kHz. Browser mic is typically 48kHz."""
    if src_rate == 16000:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * 16000 / src_rate))
    return _scipy_resample(audio, n_out).astype(np.float32)


def handle_audio_chunk(state: dict | None, audio_chunk) -> tuple:
    """Gradio .stream() handler: gate noise, accumulate speech, flush to Whisper on silence or hard cap."""
    if audio_chunk is None or get_current_session() is None:
        return state, gr.update()

    if state is None:
        state = {
            "buffer": np.array([], dtype=np.float32),
            "qa_buffer": np.array([], dtype=np.float32),
            "silence_samples": 0,
            "paused": False,
            "ambient_rms": 0.005,
            "qa_silence_samples": 0,
        }

    if state.get("paused", False):
        return state, gr.update()

    session = get_current_session()
    sample_rate, audio_data = audio_chunk

    # Normalize: mono, float32, 16kHz
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_16k = _resample_to_16k(audio_data, sample_rate)

    mode = session.current_mode()

    if mode == Mode.LECTURE:
        if session.in_conversation:
            # No lecture transcription while conversation is active
            return state, gr.update()

        chunk_rms = float(np.sqrt(np.mean(audio_16k ** 2)))
        silence_samples = state.get("silence_samples", 0)

        # Track ambient noise floor passively. Update only when the chunk is
        # below 3x current ambient (i.e., not speech), so speech events don't
        # inflate the estimate. Used by the Q&A path for adaptive silence detection.
        ambient = state.get("ambient_rms", 0.005)
        if chunk_rms < ambient * 3.0:
            ambient = (1 - _AMBIENT_ALPHA) * ambient + _AMBIENT_ALPHA * chunk_rms
            ambient = max(0.001, min(ambient, 0.1))

        if chunk_rms >= _SPEECH_ENERGY_THRESHOLD:
            # Speech chunk: accumulate and reset silence counter
            buf = np.concatenate([state["buffer"], audio_16k])
            silence_samples = 0
        else:
            # Noise/silence chunk: discard and advance silence counter
            buf = state["buffer"]
            silence_samples += len(audio_16k)

        silence_ms = silence_samples * 1000 / 16000
        buf_s = len(buf) / 16000
        should_flush = len(buf) > 0 and (
            silence_ms >= _VAD_SILENCE_MS or buf_s >= _MAX_BUFFER_S
        )
        if should_flush:
            push_audio(buf)
            buf = np.array([], dtype=np.float32)
            silence_samples = 0

        return {**state, "buffer": buf, "silence_samples": silence_samples, "ambient_rms": ambient}, poll_transcript()

    elif mode == Mode.AWAITING_QUESTION:
        qa_buf = np.concatenate([state["qa_buffer"], audio_16k])

        if session.in_conversation:
            # Conversation mode: accumulate until "Done speaking" click.
            # Safety cap so a forgotten click flushes at _CONV_MAX_TURN_S.
            if len(qa_buf) / 16000 >= _CONV_MAX_TURN_S:
                print("[CONV] hard cap reached, auto-flushing turn", flush=True)
                push_question_audio(qa_buf)
                qa_buf = np.array([], dtype=np.float32)
            return {**state, "qa_buffer": qa_buf}, gr.update()
        else:
            # Single-turn Ask AI: adaptive silence detection + hard cap.
            # Threshold calibrated against ambient level measured during lecture,
            # so a noisy room doesn't prevent end-of-question detection.
            ambient = state.get("ambient_rms", 0.005)
            adaptive_thresh = max(0.005, ambient * _AMBIENT_SPEECH_MULTIPLIER)

            chunk_rms = float(np.sqrt(np.mean(audio_16k ** 2)))
            qa_silence_samples = state.get("qa_silence_samples", 0)
            if chunk_rms < adaptive_thresh:
                qa_silence_samples += len(audio_16k)
            else:
                qa_silence_samples = 0

            buf_s = len(qa_buf) / 16000
            qa_silence_ms = qa_silence_samples * 1000 / 16000

            should_flush = (
                buf_s >= _QA_MAX_WINDOW_S
                or (buf_s >= _QA_MIN_WINDOW_S and qa_silence_ms >= _VAD_SILENCE_MS)
            )
            if should_flush:
                reason = "hard cap" if buf_s >= _QA_MAX_WINDOW_S else "silence"
                print(f"[ASK AI] auto-flush ({reason}, {buf_s:.1f}s, thresh={adaptive_thresh:.4f})", flush=True)
                push_question_audio(qa_buf)
                qa_buf = np.array([], dtype=np.float32)
                qa_silence_samples = 0

            return {**state, "qa_buffer": qa_buf, "qa_silence_samples": qa_silence_samples}, gr.update()

    else:  # Mode.PROCESSING — discard audio while handler runs
        return state, gr.update()


def stop_recording_audio(state: dict | None) -> tuple:
    """Flush remaining buffer when the browser mic recording stops."""
    session = get_current_session()
    if session is not None and state is not None:
        buf = state.get("buffer", np.array([], dtype=np.float32))
        if len(buf) > 0:
            push_audio(buf)
    empty = {
        "buffer": np.array([], dtype=np.float32),
        "qa_buffer": np.array([], dtype=np.float32),
        "silence_samples": 0,
        "paused": False,
        "ambient_rms": 0.005,
        "qa_silence_samples": 0,
    }
    return empty, poll_transcript()


def handle_pause(state: dict | None) -> tuple:
    """Flip the paused flag to True; audio accumulation stops until resume."""
    if state is None:
        state = {"buffer": np.array([], dtype=np.float32), "qa_buffer": np.array([], dtype=np.float32),
                 "silence_samples": 0, "paused": False, "ambient_rms": 0.005, "qa_silence_samples": 0}
    return {**state, "paused": True}, gr.update(interactive=False), gr.update(interactive=True)


def handle_resume(state: dict | None) -> tuple:
    """Flip the paused flag to False; audio accumulation resumes."""
    if state is None:
        state = {"buffer": np.array([], dtype=np.float32), "qa_buffer": np.array([], dtype=np.float32),
                 "silence_samples": 0, "paused": False, "ambient_rms": 0.005, "qa_silence_samples": 0}
    return {**state, "paused": False}, gr.update(interactive=True), gr.update(interactive=False)


def _on_live_gap(text: str, ts: datetime | None) -> None:
    global _live_gap_result, _live_gap_updated
    with _live_gap_lock:
        _live_gap_result = text
        _live_gap_updated = ts


def poll_live_gap() -> tuple[str, str]:
    with _live_gap_lock:
        text = _live_gap_result
        ts = _live_gap_updated
    ts_str = f"*Last updated: {ts.strftime('%H:%M:%S')}*" if ts else ""
    return text, ts_str


def _qa_handler(session) -> None:
    """Full Q&A cycle: RAG → synthesize → browser audio. Runs in a daemon thread."""
    question = session.pending_question
    print(f"[Q&A] handler started, question: {question!r}", flush=True)
    try:
        _set_status("Thinking...")
        pause_recording()
        print("[Q&A] mic paused, calling answer_question()", flush=True)
        answer, reasoning, sources = answer_question(question, session.linked_documents, session)
        print(f"[Q&A] got answer ({len(answer)} chars)", flush=True)
        session.add_qa_entry(question, answer, sources, reasoning=reasoning)
        _set_status("Speaking...")
        tts = _get_tts()
        if tts:
            print("[Q&A] synthesizing audio", flush=True)
            t0_tts = time.time()
            wav_bytes = tts.synthesize(answer)
            print(f"[Q&A] tts {time.time() - t0_tts:.1f}s", flush=True)
            audio_path = _save_qa_audio(wav_bytes)
            _set_qa_audio(audio_path)
            duration_s = len(wav_bytes) / (tts.sample_rate * 2)
            print(f"[Q&A] audio ready ({duration_s:.1f}s), mic resumes in {duration_s + 0.5:.1f}s", flush=True)
            threading.Timer(duration_s + 0.5, _resume_after_playback, args=(session,)).start()
            return
        else:
            print("[Q&A] TTS unavailable", flush=True)
    except Exception:
        traceback.print_exc()
        try:
            tts = _get_tts()
            if tts:
                wav_bytes = tts.synthesize("Sorry, I could not answer that.")
                audio_path = _save_qa_audio(wav_bytes)
                _set_qa_audio(audio_path)
                duration_s = len(wav_bytes) / (tts.sample_rate * 2)
                threading.Timer(duration_s + 0.5, _resume_after_playback, args=(session,)).start()
                return
        except Exception:
            traceback.print_exc()
    # Fallback: TTS unavailable or synthesis failed — resume immediately
    resume_recording()
    session.set_mode(Mode.LECTURE)
    _set_status("● Recording lecture")
    print("[Q&A] handler done (no audio), mode → LECTURE", flush=True)


def _conversation_handler(session, fast: bool = True) -> None:
    """Full conversation turn: RAG -> LLM -> (auto-TTS if Fast, text+button if Deep).
    Runs in a daemon thread fired by push_question_audio via session._qa_handler.
    """
    global _conv_pending_answer
    question = session.pending_question
    print(f"[CONV] handler started fast={fast} question={question!r}", flush=True)
    try:
        _set_status("● Thinking (fast)..." if fast else "● Thinking (deep)...")
        answer, reasoning, sources = answer_conversation_turn(
            question,
            session.linked_documents,
            session,
            fast=fast,
        )
        print(f"[CONV] answer {len(answer)} chars", flush=True)

        session.conversation_history.append({
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        })
        _set_conv_display(session)

        if fast:
            _set_status("● Speaking...")
            tts = _get_tts()
            if tts:
                t0 = time.time()
                wav_bytes = tts.synthesize(answer)
                print(f"[CONV] tts {time.time() - t0:.1f}s", flush=True)
                audio_path = _save_qa_audio(wav_bytes)
                _set_qa_audio(audio_path)
                duration_s = len(wav_bytes) / (tts.sample_rate * 2)
                threading.Timer(
                    duration_s + 0.5, _resume_after_conv_turn, args=(session,)
                ).start()
                return
            else:
                print("[CONV] TTS unavailable", flush=True)
                session.set_mode(Mode.LECTURE)
                _set_status("● Conversation — ready")
        else:
            # Deep: park the answer for the prof to review, enable "Speak this"
            with _conv_pending_lock:
                _conv_pending_answer = answer
            _set_status("● Answer ready — click 'Speak this' or 'End Conversation'")
            # Mode stays PROCESSING until speak_conv_answer_fn or end_conversation_fn
            return

    except Exception:
        traceback.print_exc()
        with _conv_pending_lock:
            _conv_pending_answer = ""
        session.set_mode(Mode.LECTURE)
        _set_status("● Conversation — ready")
        print("[CONV] handler error, returned to conversation idle", flush=True)


def start_recording(lecture_name: str, linked_docs: list[str]):
    global _live_gap_worker, _live_gap_result, _live_gap_updated
    try:
        # Expand any module IDs to their constituent document filenames.
        expanded_docs = expand_selection(linked_docs or [])
        session = start_live_session(
            name=lecture_name.strip(),
            linked_documents=expanded_docs,
        )
        session.register_qa_handler(_qa_handler)
        if expanded_docs:
            _set_status("● Recording lecture")
        else:
            _set_status("● Recording lecture (no source material linked — Ask AI will be disabled until you link a document)")
        with _live_gap_lock:
            _live_gap_result = (
                "Link a document to enable live gap analysis"
                if not expanded_docs else "Waiting for more content..."
            )
            _live_gap_updated = None
        _live_gap_worker = LiveGapWorker(get_current_session, _on_live_gap)
        _live_gap_worker.start()
        init_state = {
            "buffer": np.array([], dtype=np.float32),
            "qa_buffer": np.array([], dtype=np.float32),
            "silence_samples": 0,
            "paused": False,
            "ambient_rms": 0.005,
            "qa_silence_samples": 0,
        }
        return (
            gr.update(interactive=False),  # start_btn
            gr.update(interactive=True),   # stop_btn
            gr.update(interactive=True),   # ask_ai_btn
            init_state,                    # audio_state
            gr.update(interactive=True),   # pause_btn
            gr.update(interactive=False),  # resume_btn
            gr.update(interactive=True),   # conv_start_btn
            gr.update(interactive=False),  # conv_end_btn
        )
    except Exception as e:
        _set_status(f"⚠ {e}")
        return (gr.update(),) * 8


def stop_recording(state: dict | None):
    global _live_gap_worker
    # Flush any audio still in the browser buffer before closing the session
    session = get_current_session()
    if session is not None and state is not None:
        buf = state.get("buffer", np.array([], dtype=np.float32))
        if len(buf) > 0:
            push_audio(buf)
    if _live_gap_worker:
        _live_gap_worker.stop()
        _live_gap_worker = None
    stop_live_session()
    _clear_conv_state()
    _set_status("■ Stopped")
    choices = gap_dropdown_choices()
    return (
        gr.update(interactive=True),   # start_btn
        gr.update(interactive=False),  # stop_btn
        gr.update(choices=choices, value=choices[0][1] if choices else None),  # gap_dd
        gr.update(interactive=False),  # ask_ai_btn
        gr.update(interactive=False),  # ask_ai_done_btn
        gr.update(interactive=False),  # cancel_btn
        None,                          # audio_state
        gr.update(interactive=False),  # pause_btn
        gr.update(interactive=False),  # resume_btn
        gr.update(interactive=False),  # conv_start_btn
        gr.update(interactive=False),  # conv_end_btn
    )


def poll_transcript() -> str:
    session = get_current_session()
    if session is None:
        return gr.update()
    lines = []
    for seg in session.segments:
        mm, ss = divmod(int(seg["start_seconds"]), 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {seg['text']}")
    return "\n".join(lines)


def poll_status() -> str:
    return _get_status()


def poll_qa_log() -> str:
    session = get_current_session()
    if session is None:
        return gr.update()
    return session.format_qa_log()


def ask_ai_fn() -> str:
    print("[ASK AI] button clicked", flush=True)
    session = get_current_session()
    if session is None:
        print("[ASK AI] no active session", flush=True)
        return "Not recording."
    print(f"[ASK AI] current mode: {session.current_mode()}", flush=True)
    if session.current_mode() != Mode.LECTURE:
        print("[ASK AI] not in LECTURE mode — ignoring", flush=True)
        return _get_status()
    if not session.linked_documents:
        print("[ASK AI] no linked documents — refusing to enter Q&A mode")
        msg = "● No document linked — select source material first"
        _set_status(msg)
        return msg
    session.set_mode(Mode.AWAITING_QUESTION)
    _set_status("Listening for question...")
    print("[ASK AI] mode → AWAITING_QUESTION", flush=True)
    return "Listening for question..."


def cancel_qa() -> str:
    """Force mode back to LECTURE, clear pending audio, and resume the mic."""
    _set_qa_audio(None)
    resume_recording()
    session = get_current_session()
    if session:
        session.set_mode(Mode.LECTURE)
    _set_status("● Recording lecture")
    return "● Recording lecture"


def poll_qa_audio():
    path = _get_and_clear_qa_audio()
    if path:
        return gr.update(value=path)
    return gr.update()


def poll_cancel_btn() -> dict:
    session = get_current_session()
    active = session is not None and session.current_mode() != Mode.LECTURE
    return gr.update(interactive=active)


def ask_ai_done_fn(state: dict | None) -> tuple:
    """Force-flush the accumulated question buffer for single-turn Ask AI."""
    session = get_current_session()
    if session is None or session.current_mode() != Mode.AWAITING_QUESTION or session.in_conversation:
        return state, _get_status()
    qa_buf = (state or {}).get("qa_buffer", np.array([], dtype=np.float32))
    if len(qa_buf) > 0:
        print("[ASK AI] Done speaking clicked — flushing question buffer", flush=True)
        push_question_audio(qa_buf)
        new_state = {**(state or {}), "qa_buffer": np.array([], dtype=np.float32), "qa_silence_samples": 0}
        return new_state, "Processing question..."
    else:
        session.set_mode(Mode.LECTURE)
        _set_status("● Recording lecture")
        return state, "● Recording lecture"


def poll_ask_ai_done_btn() -> dict:
    session = get_current_session()
    active = (
        session is not None
        and session.current_mode() == Mode.AWAITING_QUESTION
        and not session.in_conversation
    )
    return gr.update(interactive=active)


def poll_conv_display() -> str:
    with _conv_display_lock:
        return _conv_display


def poll_conv_buttons() -> tuple:
    """Return (speak_btn, done_btn) interactive state based on conversation state."""
    session = get_current_session()
    if session is None or not session.in_conversation:
        return gr.update(interactive=False), gr.update(interactive=False)
    mode = session.current_mode()
    if mode == Mode.AWAITING_QUESTION:
        return gr.update(interactive=False), gr.update(interactive=True)
    elif mode == Mode.PROCESSING:
        return gr.update(interactive=False), gr.update(interactive=False)
    else:  # LECTURE — conversation idle, ready for next turn
        return gr.update(interactive=True), gr.update(interactive=False)


def poll_speak_text_btn() -> dict:
    with _conv_pending_lock:
        has_pending = bool(_conv_pending_answer)
    return gr.update(interactive=has_pending)


# ---------------------------------------------------------------------------
# Tab 2 — Conversation mode handlers
# ---------------------------------------------------------------------------

def start_conversation_fn() -> tuple:
    session = get_current_session()
    if session is None:
        return (gr.update(),) * 7
    session.in_conversation = True
    session.conversation_history = []
    _clear_conv_state()
    session.write_transcript_marker(
        f"--- [CONVERSATION START {datetime.now().strftime('%H:%M:%S')}] ---"
    )
    _set_status("● Conversation — ready")
    return (
        gr.update(interactive=False),  # conv_start_btn
        gr.update(interactive=True),   # conv_end_btn
        gr.update(interactive=True),   # speak_btn
        gr.update(interactive=False),  # done_btn
        gr.update(interactive=False),  # speak_text_btn
        gr.update(value=""),           # conv_display_box
        gr.update(interactive=False),  # ask_ai_btn (disabled during conversation)
    )


def end_conversation_fn() -> tuple:
    session = get_current_session()
    if session is None:
        return (gr.update(),) * 6
    session.set_mode(Mode.LECTURE)  # force-exit any stuck PROCESSING/AWAITING state
    session.in_conversation = False
    session.register_qa_handler(_qa_handler)  # restore single-turn handler
    session.write_transcript_marker(
        f"--- [CONVERSATION END {datetime.now().strftime('%H:%M:%S')}] ---"
    )
    _clear_conv_state()
    _set_status("● Recording lecture")
    return (
        gr.update(interactive=True),   # conv_start_btn
        gr.update(interactive=False),  # conv_end_btn
        gr.update(interactive=False),  # speak_btn
        gr.update(interactive=False),  # done_btn
        gr.update(interactive=False),  # speak_text_btn
        gr.update(interactive=True),   # ask_ai_btn restored
    )


def speak_btn_fn(style: str) -> tuple:
    """Arm capture for one conversation turn and register the appropriate handler."""
    session = get_current_session()
    if session is None or not session.in_conversation:
        return gr.update(interactive=False), gr.update(interactive=False)
    fast = (style == "Fast")
    session.register_qa_handler(lambda s, f=fast: _conversation_handler(s, fast=f))
    session.set_mode(Mode.AWAITING_QUESTION)
    _set_status("● Listening — speak your turn, then click Done speaking")
    return (
        gr.update(interactive=False),  # speak_btn
        gr.update(interactive=True),   # done_btn
    )


def done_speaking_fn(state: dict | None) -> tuple:
    """Flush qa_buffer immediately, regardless of how long the turn was."""
    session = get_current_session()
    qa_buf = (state or {}).get("qa_buffer", np.array([], dtype=np.float32))
    if session is not None and len(qa_buf) > 0:
        push_question_audio(qa_buf)
    elif session is not None:
        # Empty buffer — nothing to transcribe, return to conversation idle
        session.set_mode(Mode.LECTURE)
        _set_status("● Conversation — ready")
    new_state = {**state, "qa_buffer": np.array([], dtype=np.float32), "qa_silence_samples": 0} if state else state
    return (
        new_state,
        gr.update(interactive=False),  # speak_btn (poll will re-enable after PROCESSING)
        gr.update(interactive=False),  # done_btn
    )


def speak_conv_answer_fn() -> dict:
    """Synthesize and play the pending Deep mode answer."""
    global _conv_pending_answer
    with _conv_pending_lock:
        answer = _conv_pending_answer
        _conv_pending_answer = ""
    if not answer:
        return gr.update(interactive=False)
    session = get_current_session()
    tts = _get_tts()
    if tts and session:
        t0 = time.time()
        wav_bytes = tts.synthesize(answer)
        print(f"[CONV] speak-this tts {time.time() - t0:.1f}s", flush=True)
        audio_path = _save_qa_audio(wav_bytes)
        _set_qa_audio(audio_path)
        duration_s = len(wav_bytes) / (tts.sample_rate * 2)
        threading.Timer(duration_s + 0.5, _resume_after_conv_turn, args=(session,)).start()
    return gr.update(interactive=False)


# ---------------------------------------------------------------------------
# Tab 3 — Add Materials
# ---------------------------------------------------------------------------

def process_uploads(files):
    if not files:
        yield "No files selected."
        return
    DOCS_DIR.mkdir(exist_ok=True)
    LECTURES_DIR.mkdir(exist_ok=True)
    log = ""
    for f in files:
        src = Path(f.name)
        suffix = src.suffix.lower()
        if suffix in DOC_EXTS:
            dest = DOCS_DIR / src.name
            shutil.copy2(src, dest)
            for msg in ingest_paths([dest]):
                log += msg + "\n"
                yield log
        elif suffix in AUDIO_EXTS:
            dest = LECTURES_DIR / src.name
            shutil.copy2(src, dest)
            for msg in transcribe_file(dest):
                log += msg + "\n"
                yield log
        else:
            log += f"Skipped {src.name} (unsupported type)\n"
            yield log
    log += "\nAll done."
    yield log


# ---------------------------------------------------------------------------
# Tab 4 — Gaps Analysis
# ---------------------------------------------------------------------------

def run_gaps(session_choice: str, use_latest: bool, show_thinking: bool):
    try:
        session_id = resolve_session_id(
            session_id=None if use_latest else (session_choice or None),
            latest=use_latest,
        )
    except ValueError as e:
        yield str(e)
        return
    result = ""
    for token in gaps_stream(session_id, show_thinking):
        result += token
        yield result


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    _init_qa_audio_dir()
    gap_choices = gap_dropdown_choices()
    doc_choices = doc_dropdown_choices()

    with gr.Blocks(title="Prof AI") as demo:
        gr.HTML(HEADER_HTML)

        # ── Tab 1: Chat ──────────────────────────────────────────────────
        with gr.Tab("Chat") as chat_tab:
            # Declare output components with render=False so they can be
            # passed to ChatInterface but rendered below it in the layout.
            docs_md = gr.Markdown(
                "*Ask a question to see document sources.*", render=False
            )
            literature_html = gr.HTML(
                "<p style='color:var(--pa-muted); font-style:italic;'>"
                "Select a literature source to include recent papers.</p>",
                render=False,
            )

            with gr.Accordion("Search scope and literature sources", open=False):
                with gr.Row():
                    scope_dd = gr.Dropdown(choices=scope_choices(), value="All material",
                                           label="Scope queries to", scale=1)
                    doc_filter_dd = gr.Dropdown(
                        choices=unified_source_choices(), multiselect=True,
                        label="Search in (leave empty for all)", scale=3,
                    )
                with gr.Row():
                    lit_sources_cb = gr.CheckboxGroup(
                        choices=list(_LIT_SOURCE_LABELS.keys()),
                        label="Literature sources",
                        value=[],
                        scale=3,
                    )
                    lit_max_num = gr.Number(
                        value=5, minimum=1, maximum=10, precision=0,
                        label="Max papers per source", scale=0, min_width=180,
                    )
                gr.Markdown(
                    "*Select one or more sources. Combining Semantic Scholar and OpenAlex gives "
                    "the widest cross-disciplinary coverage. PubMed is best for biomedical and "
                    "neuroscience questions.*"
                )

            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[scope_dd, doc_filter_dd, lit_sources_cb, lit_max_num],
                additional_outputs=[docs_md, literature_html],
                autoscroll=True,
            )

            # Sources accordion — rendered here so it appears below the chat
            with gr.Accordion("Sources", open=False):
                with gr.Tabs():
                    with gr.Tab("Documents"):
                        docs_md.render()
                    with gr.Tab("Recent Literature"):
                        literature_html.render()

            chat_tab.select(
                fn=lambda: gr.update(choices=unified_source_choices()),
                outputs=[doc_filter_dd],
            )

        # ── Tab 2: Live Lecture ──────────────────────────────────────────
        with gr.Tab("Live Lecture"):
            # Top controls
            with gr.Group():
                with gr.Row():
                    lecture_name_box = gr.Textbox(label="Lecture name (optional)",
                                                  placeholder="e.g. Neuro Week 5", scale=2)
                    linked_docs_dd = gr.Dropdown(
                        choices=unified_source_choices(), multiselect=True,
                        label="Source material for this lecture", scale=3,
                    )
                    refresh_docs_btn = gr.Button("↻", scale=0, min_width=40)
                status_box = gr.Textbox(value="■ Stopped", label="Status",
                                        interactive=False, elem_id="pa-status")
            with gr.Row():
                start_btn = gr.Button("▶ Start Lecture", variant="primary")
                stop_btn = gr.Button("■ Stop Lecture", variant="stop", interactive=False)
                pause_btn = gr.Button("⏸ Pause Mic", interactive=False)
                resume_btn = gr.Button("▶ Resume Mic", interactive=False)
                ask_ai_btn = gr.Button("Ask AI", interactive=False)
                ask_ai_done_btn = gr.Button("✓ Done Speaking", interactive=False)
                cancel_btn = gr.Button("✕ Cancel", interactive=False)
            mic_audio = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Microphone (click record, then Start Lecture)",
            )
            audio_state = gr.State(value=None)

            # Side-by-side: transcript | live gap analysis
            with gr.Row():
                with gr.Column():
                    transcript_box = gr.Textbox(label="Live Transcript", lines=20,
                                                interactive=False, autoscroll=True,
                                                elem_id="pa-transcript")
                with gr.Column():
                    gap_box = gr.Textbox(label="Live Gap Analysis", lines=19,
                                         interactive=False, elem_id="pa-gap")
                    gap_ts_md = gr.Markdown("")

            # Q&A history (collapsed by default)
            with gr.Accordion("Q&A History", open=False):
                qa_log_box = gr.Textbox(label="", lines=8,
                                        interactive=False, autoscroll=True,
                                        elem_id="pa-qa-log")

            qa_audio = gr.Audio(
                label="AI Response",
                autoplay=True,
                visible=True,
                streaming=False,
                interactive=False,
            )

            with gr.Accordion("Conversation Mode", open=False):
                response_style = gr.Radio(
                    choices=["Fast", "Deep"],
                    value="Fast",
                    label="Response style  (Fast: short spoken answer, think off  |  Deep: full answer with reasoning, shown as text)",
                )
                with gr.Row():
                    conv_start_btn = gr.Button("▶ Start Conversation", interactive=False)
                    conv_end_btn = gr.Button("■ End Conversation", interactive=False)
                with gr.Row():
                    speak_btn = gr.Button("🎙 Speak to AI", interactive=False)
                    done_btn = gr.Button("✓ Done speaking", interactive=False)
                conv_display_box = gr.Textbox(
                    label="Conversation",
                    lines=8,
                    interactive=False,
                    autoscroll=True,
                    elem_id="pa-conv",
                )
                speak_text_btn = gr.Button("Speak this", interactive=False)

            # Timers
            timer = gr.Timer(value=2)
            timer.tick(fn=poll_transcript, outputs=[transcript_box])
            timer.tick(fn=poll_status, outputs=[status_box])
            timer.tick(fn=poll_qa_log, outputs=[qa_log_box])
            timer.tick(fn=poll_cancel_btn, outputs=[cancel_btn])
            timer.tick(fn=poll_ask_ai_done_btn, outputs=[ask_ai_done_btn])
            timer.tick(fn=poll_qa_audio, outputs=[qa_audio])
            timer.tick(fn=poll_conv_display, outputs=[conv_display_box])
            timer.tick(fn=poll_conv_buttons, outputs=[speak_btn, done_btn])
            timer.tick(fn=poll_speak_text_btn, outputs=[speak_text_btn])
            gap_timer = gr.Timer(value=5)
            gap_timer.tick(fn=poll_live_gap, outputs=[gap_box, gap_ts_md])

            # Intra-tab events
            refresh_docs_btn.click(
                fn=lambda: gr.update(choices=unified_source_choices()),
                outputs=[linked_docs_dd],
            )
            ask_ai_btn.click(fn=ask_ai_fn, outputs=[status_box])
            ask_ai_done_btn.click(
                fn=ask_ai_done_fn,
                inputs=[audio_state],
                outputs=[audio_state, status_box],
            )
            cancel_btn.click(fn=cancel_qa, outputs=[status_box])
            pause_btn.click(
                fn=handle_pause,
                inputs=[audio_state],
                outputs=[audio_state, pause_btn, resume_btn],
            )
            resume_btn.click(
                fn=handle_resume,
                inputs=[audio_state],
                outputs=[audio_state, pause_btn, resume_btn],
            )
            conv_start_btn.click(
                fn=start_conversation_fn,
                outputs=[conv_start_btn, conv_end_btn, speak_btn, done_btn,
                         speak_text_btn, conv_display_box, ask_ai_btn],
            )
            conv_end_btn.click(
                fn=end_conversation_fn,
                outputs=[conv_start_btn, conv_end_btn, speak_btn, done_btn,
                         speak_text_btn, ask_ai_btn],
            )
            speak_btn.click(
                fn=speak_btn_fn,
                inputs=[response_style],
                outputs=[speak_btn, done_btn],
            )
            done_btn.click(
                fn=done_speaking_fn,
                inputs=[audio_state],
                outputs=[audio_state, speak_btn, done_btn],
            )
            speak_text_btn.click(
                fn=speak_conv_answer_fn,
                outputs=[speak_text_btn],
            )
            mic_audio.stream(
                fn=handle_audio_chunk,
                inputs=[audio_state, mic_audio],
                outputs=[audio_state, transcript_box],
            )
            mic_audio.stop_recording(
                fn=stop_recording_audio,
                inputs=[audio_state],
                outputs=[audio_state, transcript_box],
            )

        # ── Tab 3: Add Materials ─────────────────────────────────────────
        with gr.Tab("Add Materials"):
            upload = gr.File(
                label="Upload files (.pdf, .txt, .docx, .pptx, .mp3, .wav, .m4a, .flac)",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".docx", ".pptx", ".mp3", ".wav", ".m4a", ".flac"],
            )
            process_btn = gr.Button("Process", variant="primary")
            process_log = gr.Textbox(label="Progress log", lines=10, interactive=False)
            process_btn.click(fn=process_uploads, inputs=[upload], outputs=[process_log])

        # ── Tab 4: Modules ───────────────────────────────────────────────
        with gr.Tab("Modules") as modules_tab:
            # Hidden bridge: JS writes the drag payload to #drag_state, then clicks
            # #drag_trigger to fire on_drag_change through Gradio's real event system.
            drag_state = gr.Textbox(visible=True, elem_id="drag_state", label="", show_label=False, elem_classes=["pma-hidden-bridge"])
            drag_trigger = gr.Button("trigger", elem_id="drag_trigger", elem_classes=["pma-hidden-bridge"])

            # Drag-drop three-pane panel (left=class+module tree, mid=docs, right=library)
            drag_panel_html = gr.HTML(_render_drag_panel())

            modules_status_tb = gr.Textbox(label="", interactive=False, lines=1)

            with gr.Row(equal_height=False):
                # ── Module controls ──────────────────────────────────────
                with gr.Column():
                    gr.Markdown("**Modules**")
                    with gr.Row():
                        new_mod_tb = gr.Textbox(
                            label="New module name",
                            placeholder="e.g. Week 3 – Synaptic Plasticity",
                            scale=3,
                        )
                        create_mod_btn = gr.Button("Create", scale=0, min_width=80)
                    with gr.Row():
                        module_dd = gr.Dropdown(
                            choices=module_dd_choices(),
                            label="Select module (for rename/delete)",
                            scale=3,
                        )
                        mod_refresh_btn = gr.Button("↻", scale=0, min_width=40)
                    with gr.Row():
                        rename_mod_tb = gr.Textbox(
                            label="Rename selected to", placeholder="New name", scale=3,
                        )
                        rename_mod_btn = gr.Button("Rename", scale=0, min_width=80)
                    with gr.Row():
                        delete_mod_btn = gr.Button("Delete Selected", variant="stop",
                                                   scale=0, min_width=120)
                        confirm_delete_btn = gr.Button("Confirm Delete", variant="stop",
                                                       interactive=False, scale=0, min_width=120)

                # ── Class controls ───────────────────────────────────────
                with gr.Column():
                    gr.Markdown("**Classes** — drag modules between class folders in the panel above")
                    with gr.Row():
                        new_class_tb = gr.Textbox(
                            label="New class name",
                            placeholder="e.g. COMM 3310",
                            scale=3,
                        )
                        create_class_btn = gr.Button("Create Class", scale=0, min_width=100)
                    with gr.Row():
                        class_dd = gr.Dropdown(
                            choices=class_dd_choices(),
                            label="Select class (for rename/delete)",
                            scale=3,
                        )
                    with gr.Row():
                        rename_class_tb = gr.Textbox(
                            label="Rename selected class to", placeholder="New name", scale=3,
                        )
                        rename_class_btn = gr.Button("Rename", scale=0, min_width=80)
                    with gr.Row():
                        delete_class_btn = gr.Button("Delete Class", variant="stop",
                                                     scale=0, min_width=120)
                        confirm_delete_class_btn = gr.Button("Confirm Delete Class", variant="stop",
                                                             interactive=False, scale=0, min_width=140)
                    classes_status_tb = gr.Textbox(label="", interactive=False, lines=1)

            # ── Event wiring ─────────────────────────────────────────────
            drag_trigger.click(
                fn=on_drag_change,
                inputs=[drag_state],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html],
                js="() => { const el=document.querySelector('#drag_state textarea'); return el ? el.value : ''; }",
            )
            mod_refresh_btn.click(fn=refresh_modules_tab, outputs=[module_dd])
            modules_tab.select(fn=refresh_modules_tab, outputs=[module_dd])
            create_mod_btn.click(
                fn=create_module_fn,
                inputs=[new_mod_tb],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html,
                         new_mod_tb, modules_status_tb],
            )
            rename_mod_btn.click(
                fn=rename_module_fn,
                inputs=[module_dd, rename_mod_tb],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html,
                         modules_status_tb],
            )
            delete_mod_btn.click(
                fn=delete_module_stage1,
                inputs=[module_dd],
                outputs=[confirm_delete_btn, modules_status_tb],
            )
            confirm_delete_btn.click(
                fn=delete_module_stage2,
                inputs=[module_dd],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html,
                         confirm_delete_btn, modules_status_tb],
            )
            create_class_btn.click(
                fn=create_class_fn,
                inputs=[new_class_tb],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html,
                         class_dd, new_class_tb, classes_status_tb],
            )
            rename_class_btn.click(
                fn=rename_class_fn,
                inputs=[class_dd, rename_class_tb],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html,
                         class_dd, classes_status_tb],
            )
            delete_class_btn.click(
                fn=delete_class_stage1,
                inputs=[class_dd],
                outputs=[confirm_delete_class_btn, classes_status_tb],
            )
            confirm_delete_class_btn.click(
                fn=delete_class_stage2,
                inputs=[class_dd],
                outputs=[module_dd, linked_docs_dd, doc_filter_dd, drag_panel_html,
                         class_dd, confirm_delete_class_btn, classes_status_tb],
            )

        # ── Tab 5: Gaps Analysis ─────────────────────────────────────────
        with gr.Tab("Gaps Analysis") as gaps_tab:
            with gr.Row():
                gap_dd = gr.Dropdown(
                    choices=gap_choices,
                    value=gap_choices[0][1] if gap_choices else None,
                    label="Session", scale=3,
                )
                gap_refresh_btn = gr.Button("↻ Refresh", scale=0, min_width=90)
                use_latest_cb = gr.Checkbox(label="Use latest session", scale=1)
            linked_docs_label = gr.Textbox(
                value=linked_docs_info(gap_choices[0][1] if gap_choices else None)[0],
                label="Document scope", interactive=False, lines=1,
            )
            with gr.Row():
                linked_docs_edit_dd = gr.Dropdown(
                    choices=doc_choices,
                    value=linked_docs_info(gap_choices[0][1] if gap_choices else None)[1],
                    multiselect=True, label="Linked documents (edit & save)", scale=3,
                )
                save_links_btn = gr.Button("Save", scale=0, min_width=60)
            show_thinking_cb = gr.Checkbox(label="Show reasoning trace")
            gaps_btn = gr.Button("Find Gaps", variant="primary")
            gaps_out = gr.Textbox(label="Gap analysis", lines=20, interactive=False)

            # Intra-tab events
            gaps_btn.click(fn=run_gaps,
                           inputs=[gap_dd, use_latest_cb, show_thinking_cb],
                           outputs=[gaps_out])
            gap_dd.change(fn=linked_docs_info, inputs=[gap_dd],
                          outputs=[linked_docs_label, linked_docs_edit_dd])
            save_links_btn.click(fn=save_linked_docs_fn,
                                 inputs=[gap_dd, linked_docs_edit_dd],
                                 outputs=[linked_docs_label])
            gap_refresh_btn.click(fn=refresh_gaps_tab,
                                  outputs=[gap_dd, linked_docs_label, linked_docs_edit_dd])
            gaps_tab.select(fn=refresh_gaps_tab,
                            outputs=[gap_dd, linked_docs_label, linked_docs_edit_dd])

        # ── Cross-tab events ──────────────────────────────────────────────
        start_btn.click(
            fn=start_recording,
            inputs=[lecture_name_box, linked_docs_dd],
            outputs=[start_btn, stop_btn, ask_ai_btn, audio_state, pause_btn, resume_btn,
                     conv_start_btn, conv_end_btn],
        )
        stop_btn.click(
            fn=stop_recording,
            inputs=[audio_state],
            outputs=[start_btn, stop_btn, gap_dd, ask_ai_btn, ask_ai_done_btn, cancel_btn,
                     audio_state, pause_btn, resume_btn, conv_start_btn, conv_end_btn],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=THEME,
        css=CSS,
        js=FORCE_DARK_JS,
        share=True,
        auth=("prof", "password"),
    )