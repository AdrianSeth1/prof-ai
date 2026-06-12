"""
ui_theme.py — visual identity for the Prof AI Gradio app.

Everything visual lives here: the Gradio theme object, the custom CSS
(including the --pa-* variables that the inline HTML in app.py references),
the JS snippet that forces dark mode, and the header HTML.

app.py imports THEME, CSS, FORCE_DARK_JS, HEADER_HTML and passes the first
three to launch(). In Gradio 6, theme/css/js are launch() parameters, not
Blocks() parameters.

No external font fetches. The app must look right offline, so we use system
font stacks only.
"""

import gradio as gr

_FONT = ["ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"]
_FONT_MONO = ["ui-monospace", "Cascadia Mono", "Consolas", "Menlo", "monospace"]

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=_FONT,
    font_mono=_FONT_MONO,
).set(
    # surfaces
    body_background_fill_dark="#0d1017",
    background_fill_primary_dark="#151a24",
    background_fill_secondary_dark="#1a202c",
    block_background_fill_dark="#151a24",
    panel_background_fill_dark="#1a202c",
    # borders
    block_border_color_dark="#2a3242",
    border_color_primary_dark="#2a3242",
    input_border_color_dark="#39435a",
    # text
    body_text_color_dark="#e7ebf3",
    body_text_color_subdued_dark="#8a94a8",
    block_title_text_color_dark="#9fa9bb",
    block_label_text_color_dark="#8a94a8",
    link_text_color_dark="#5eead4",
    # inputs and controls
    input_background_fill_dark="#1a202c",
    checkbox_background_color_dark="#1a202c",
    slider_color_dark="#34d399",
    color_accent_soft_dark="rgba(52,211,153,.14)",
    # buttons
    button_primary_background_fill_dark="#0e9f6e",
    button_primary_background_fill_hover_dark="#10b981",
    button_secondary_background_fill_dark="#232b3a",
    button_secondary_background_fill_hover_dark="#2c3546",
    button_cancel_background_fill_dark="#7f1d1d",
    # tables
    table_even_background_fill_dark="#151a24",
    table_odd_background_fill_dark="#1a202c",
    block_shadow_dark="0 1px 3px rgba(0,0,0,.35)",
)

# Reload once with ?__theme=dark so Gradio renders its dark palette.
# This is the reliable way to force dark mode across Gradio versions.
FORCE_DARK_JS = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark');
    window.location.replace(u.href);
  }
}
"""

CSS = """
/* ---- palette used by inline HTML in app.py (drag panel, literature cards) ---- */
:root {
  --pa-bg-1: #151a24;            /* card / panel surface        (was #fff)    */
  --pa-bg-2: #1a202c;            /* subtle alt surface          (was #f9fafb) */
  --pa-bg-3: #232b3a;            /* section headers, row rules  (was #f3f4f6) */
  --pa-border: #2a3242;          /* main borders                (was #e5e7eb) */
  --pa-border-2: #39435a;        /* input borders               (was #d1d5db) */
  --pa-border-faint: #222937;    /* faint rules                 (was #f0f0f0) */
  --pa-text: #e7ebf3;            /* primary text                (was #111827) */
  --pa-text-2: #c8d0dd;          /* secondary text              (was #374151) */
  --pa-text-3: #9fa9bb;          /* tertiary text               (was #4b5563) */
  --pa-muted: #8a94a8;           /* muted text                  (was #6b7280) */
  --pa-muted-2: #67718a;         /* extra-muted text            (was #9ca3af) */
  --pa-accent: #34d399;          /* accent                      (was #3b82f6) */
  --pa-accent-border: rgba(52,211,153,.45);   /* (was #93c5fd) */
  --pa-accent-bg: rgba(52,211,153,.16);       /* (was #dbeafe) */
  --pa-accent-bg-soft: rgba(52,211,153,.07);  /* (was #eff6ff) */
  --pa-link: #5eead4;            /* links                       (was #1d4ed8) */
}

/* ---- layout ---- */
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---- header ---- */
#pa-header { display: flex; align-items: center; gap: 14px; padding: 16px 4px 4px; }
#pa-header .pa-logo {
  width: 40px; height: 40px; border-radius: 11px; flex-shrink: 0;
  background: linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
  display: flex; align-items: center; justify-content: center;
  font: 800 19px/1 ui-monospace, Consolas, monospace; color: #06251c;
}
#pa-header .pa-title {
  font-size: 20px; font-weight: 700; letter-spacing: -0.01em;
  color: var(--pa-text); line-height: 1.1;
}
#pa-header .pa-sub { font-size: 12.5px; color: var(--pa-muted); margin-top: 2px; }

/* ---- live lecture panes: monospace, readable ---- */
#pa-transcript textarea, #pa-gap textarea, #pa-qa-log textarea, #pa-conv textarea {
  font-family: ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace;
  font-size: 12.8px; line-height: 1.6;
}
#pa-status textarea {
  font-weight: 700; letter-spacing: .03em;
  color: var(--pa-accent) !important;
}

/* ---- drag-and-drop bridge: keep in DOM, move off screen ----
   display:none would be fine for the programmatic .click(), but offscreen
   positioning is the safest cross-browser way to keep it interactable. */
.pma-hidden-bridge {
  position: absolute !important; left: -9999px !important; top: 0 !important;
  width: 1px !important; height: 1px !important;
  min-width: 0 !important; overflow: hidden !important;
}

/* ---- scrollbars ---- */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb { background: #2c3546; border-radius: 5px; }
*::-webkit-scrollbar-thumb:hover { background: #39435a; }
*::-webkit-scrollbar-track { background: transparent; }
"""

HEADER_HTML = """
<div id="pa-header">
  <div class="pa-logo">P</div>
  <div>
    <div class="pa-title">Prof AI</div>
    <div class="pa-sub">Lecture capture &middot; voice Q&amp;A &middot; research assistant</div>
  </div>
</div>
"""
