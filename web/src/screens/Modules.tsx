import { useCallback, useEffect, useRef, useState } from 'react'
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  useSensor,
  useSensors,
  closestCenter,
  useDraggable,
  useDroppable,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
  arrayMove,
} from '@dnd-kit/sortable'

// ── Types ─────────────────────────────────────────────────────────

interface ModuleData { id: string; name: string; documents: string[]; class_id: string | null }
interface ClassGroup { class_id: string | null; name: string; modules: ModuleData[] }
interface BoardData { classes: ClassGroup[]; library: string[] }
interface Props { onToast: (title: string, desc?: string) => void }

type DragType = 'module' | 'mod-doc' | 'lib-doc'
interface ActiveDrag { type: DragType; label: string }

// ── Helpers ───────────────────────────────────────────────────────

function xform(t: { x: number; y: number } | null): string | undefined {
  return t ? `translate3d(${t.x}px,${t.y}px,0)` : undefined
}

async function apiFetch(url: string, method: string, body?: object): Promise<unknown> {
  const r = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail ?? `HTTP ${r.status}`)
  }
  return r.json()
}

// ── Drag grip icon ────────────────────────────────────────────────

function GripIcon({ color = '#5a5e66' }: { color?: string }) {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none"
      stroke={color} strokeWidth="1.5" strokeLinecap="round" style={{ flexShrink: 0 }}>
      <path d="M6 4h.01M10 4h.01M6 8h.01M10 8h.01M6 12h.01M10 12h.01"/>
    </svg>
  )
}

function FileIcon({ color = '#8a8f98' }: { color?: string }) {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none"
      stroke={color} strokeWidth="1.4" style={{ flexShrink: 0 }}>
      <path d="M4 2h5l3 3v9H4z"/>
      <path d="M9 2v3h3"/>
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 16 16" fill="none"
      stroke="var(--ok)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"
      style={{ flexShrink: 0 }}>
      <path d="M3 8.5l3.2 3.2L13 4.5"/>
    </svg>
  )
}

// ── ClassRow — droppable for module assignment ────────────────────

function ClassRow({ group, isSelected, moduleCount, onClick }: {
  group: ClassGroup
  isSelected: boolean
  moduleCount: number
  onClick: () => void
}) {
  const dropId = `class:${group.class_id ?? 'null'}`
  const { setNodeRef, isOver } = useDroppable({
    id: dropId,
    data: { type: 'class-drop', classId: group.class_id },
  })

  return (
    <div
      ref={setNodeRef}
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '10px',
        borderRadius: 7,
        cursor: 'pointer',
        background: isSelected
          ? 'rgba(94,106,210,0.10)'
          : isOver ? 'rgba(94,106,210,0.06)' : 'transparent',
        border: `1px solid ${isSelected || isOver ? 'rgba(94,106,210,0.28)' : 'transparent'}`,
        transition: 'background 140ms ease, border-color 140ms ease',
      }}
    >
      <div>
        <div style={{
          fontSize: 12.5,
          fontWeight: 500,
          color: isSelected ? 'var(--text)' : 'var(--text-soft)',
        }}>
          {group.name}
        </div>
        <div style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 10,
          marginTop: 2,
          color: isSelected ? 'var(--accent-light)' : 'var(--text-faint)',
        }}>
          {moduleCount} module{moduleCount !== 1 ? 's' : ''}
          {isSelected ? ' · selected' : ''}
        </div>
      </div>
    </div>
  )
}

// ── ModuleRow — draggable to a class ─────────────────────────────

function ModuleRow({ module, index, isSelected, onClick }: {
  module: ModuleData
  index: number
  isSelected: boolean
  onClick: () => void
}) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `module:${module.id}`,
    data: { type: 'module' as DragType, moduleId: module.id, label: module.name },
  })

  return (
    <div
      ref={setNodeRef}
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        padding: '9px',
        borderRadius: 7,
        cursor: isDragging ? 'grabbing' : 'grab',
        opacity: isDragging ? 0.35 : 1,
        background: isSelected ? 'rgba(94,106,210,0.10)' : 'transparent',
        border: `1px solid ${isSelected ? 'rgba(94,106,210,0.28)' : 'transparent'}`,
        transition: 'background 140ms ease, border-color 140ms ease',
        userSelect: 'none',
      }}
      {...attributes}
      {...listeners}
    >
      <GripIcon color={isSelected ? '#8b95ea' : '#5a5e66'} />
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 10,
        color: isSelected ? 'var(--accent-light)' : 'var(--text-faint)',
        minWidth: 18,
      }}>
        {String(index + 1).padStart(2, '0')}
      </span>
      <span style={{
        fontSize: 12.5,
        flex: 1,
        color: isSelected ? 'var(--text)' : 'var(--text-soft)',
        fontWeight: isSelected ? 500 : 400,
      }}>
        {module.name}
      </span>
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 10,
        color: isSelected ? 'var(--accent-light)' : 'var(--text-faint)',
      }}>
        {module.documents.length}
      </span>
    </div>
  )
}

// ── ModDocRow — sortable doc inside a module ──────────────────────

function ModDocRow({ filename, index, onRemove }: {
  filename: string
  index: number
  onRemove: (f: string) => void
}) {
  const {
    attributes, listeners, setNodeRef,
    transform, transition, isDragging,
  } = useSortable({
    id: `moddoc:${filename}`,
    data: { type: 'mod-doc' as DragType, filename },
  })

  return (
    <div
      ref={setNodeRef}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        padding: 9,
        borderRadius: 7,
        border: '1px solid rgba(255,255,255,0.06)',
        background: 'var(--surface-2)',
        cursor: isDragging ? 'grabbing' : 'grab',
        opacity: isDragging ? 0.35 : 1,
        transform: xform(transform),
        transition,
        userSelect: 'none',
      }}
      {...attributes}
      {...listeners}
    >
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 10,
        color: 'var(--text-faint)',
        minWidth: 14,
      }}>
        {index + 1}
      </span>
      <FileIcon />
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 11,
        flex: 1,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        color: 'var(--text-soft)',
      }}>
        {filename}
      </span>
      <button
        onPointerDown={e => e.stopPropagation()}
        onClick={e => { e.stopPropagation(); onRemove(filename) }}
        title="Remove from module"
        style={{
          background: 'none',
          border: 'none',
          color: 'var(--text-faint)',
          cursor: 'pointer',
          padding: 2,
          lineHeight: 1,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none"
          stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <path d="M4 4l8 8M12 4l-8 8"/>
        </svg>
      </button>
    </div>
  )
}

// ── LibDocRow — draggable from library ───────────────────────────

function LibDocRow({ filename, inModule }: { filename: string; inModule: boolean }) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `libdoc:${filename}`,
    data: { type: 'lib-doc' as DragType, filename },
  })

  return (
    <div
      ref={setNodeRef}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '7px 8px',
        borderRadius: 6,
        cursor: isDragging ? 'grabbing' : 'grab',
        opacity: isDragging ? 0.35 : 1,
        userSelect: 'none',
      }}
      {...attributes}
      {...listeners}
    >
      <FileIcon color={inModule ? '#8a8f98' : '#c4c8cf'} />
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 11,
        flex: 1,
        color: inModule ? 'var(--text-muted)' : 'var(--text-soft)',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>
        {filename}
      </span>
      {inModule && <CheckIcon />}
    </div>
  )
}

// ── Module-docs drop zone ─────────────────────────────────────────

function ModuleDocsZone({ children, highlightDrop }: { children: React.ReactNode; highlightDrop: boolean }) {
  const { setNodeRef, isOver } = useDroppable({
    id: 'module-docs-zone',
    data: { type: 'module-docs-zone' },
  })

  const showHighlight = highlightDrop && isOver

  return (
    <div
      ref={setNodeRef}
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: 5,
        minHeight: 340,
        background: 'var(--surface)',
        border: `1px solid ${showHighlight ? 'rgba(94,106,210,0.40)' : 'rgba(255,255,255,0.06)'}`,
        borderRadius: 9,
        padding: 6,
        transition: 'border-color 140ms ease',
      }}
    >
      {children}
    </div>
  )
}

// ── ColLabel ─────────────────────────────────────────────────────

function ColLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      padding: '6px 8px 8px',
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: 10,
      color: 'var(--text-muted)',
      letterSpacing: '0.05em',
      gap: 6,
    }}>
      {children}
    </div>
  )
}

// ── Inline create/rename form ─────────────────────────────────────

function CreateRow({ placeholder, onSubmit, error }: {
  placeholder: string
  onSubmit: (name: string) => Promise<void>
  error?: string | null
}) {
  const [val, setVal] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit() {
    if (!val.trim() || busy) return
    setBusy(true)
    try { await onSubmit(val.trim()); setVal('') }
    finally { setBusy(false) }
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={val}
          onChange={e => setVal(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
          placeholder={placeholder}
          style={{
            flex: 1, height: 34,
            background: 'var(--surface)',
            border: '1px solid var(--line-input)',
            borderRadius: 8,
            outline: 'none',
            color: 'var(--text)',
            fontFamily: 'inherit',
            fontSize: 12.5,
            padding: '0 11px',
          }}
        />
        <button
          onClick={submit}
          disabled={busy || !val.trim()}
          style={{
            height: 34, padding: '0 15px',
            borderRadius: 8, border: 'none',
            background: val.trim() ? 'var(--accent)' : 'rgba(94,106,210,0.3)',
            color: '#fff', fontSize: 12.5, fontWeight: 500,
            cursor: val.trim() ? 'pointer' : 'default',
            fontFamily: 'inherit',
            transition: 'background 140ms ease',
          }}
        >
          Create
        </button>
      </div>
      {error && (
        <div style={{ fontSize: 11.5, color: 'var(--rec-text)', marginTop: 5 }}>{error}</div>
      )}
    </div>
  )
}

// ── Ghost overlay (DragOverlay content) ──────────────────────────

function DragGhost({ drag }: { drag: ActiveDrag }) {
  return (
    <div style={{
      padding: '8px 12px',
      borderRadius: 7,
      background: 'var(--surface-pop)',
      border: '1px solid var(--accent-line)',
      fontSize: 12.5,
      color: 'var(--accent-light)',
      fontFamily: drag.type === 'mod-doc' || drag.type === 'lib-doc'
        ? '"JetBrains Mono", monospace'
        : 'inherit',
      boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
      pointerEvents: 'none',
      maxWidth: 280,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    }}>
      {drag.label}
    </div>
  )
}

// ── Ghost button helpers ──────────────────────────────────────────

function GhostBtn({ label, onClick, danger = false }: { label: string; onClick: () => void; danger?: boolean }) {
  return (
    <button
      onClick={onClick}
      style={{
        height: 34, padding: '0 13px',
        borderRadius: 8, fontSize: 12.5,
        cursor: 'pointer', fontFamily: 'inherit',
        background: danger ? 'rgba(239,77,86,0.07)' : 'transparent',
        border: `1px solid ${danger ? 'rgba(239,77,86,0.30)' : 'var(--line-input)'}`,
        color: danger ? 'var(--rec-text)' : 'var(--text-soft)',
        transition: 'background 140ms ease, color 140ms ease',
      }}
    >
      {label}
    </button>
  )
}

function AccentBtn({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        height: 34, padding: '0 15px',
        borderRadius: 8, border: 'none',
        background: 'var(--accent)', color: '#fff',
        fontSize: 12.5, fontWeight: 500,
        cursor: 'pointer', fontFamily: 'inherit',
      }}
    >
      {label}
    </button>
  )
}

// ── Empty state ───────────────────────────────────────────────────

function EmptySlot({ text }: { text: string }) {
  return (
    <div style={{
      marginTop: 'auto',
      textAlign: 'center',
      fontFamily: '"JetBrains Mono", monospace',
      fontSize: 10,
      color: 'var(--text-ghost)',
      border: '1px dashed rgba(255,255,255,0.08)',
      borderRadius: 7,
      padding: 10,
    }}>
      {text}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────

export default function Modules({ onToast }: Props) {
  const [board,       setBoard]       = useState<BoardData | null>(null)
  const [loading,     setLoading]     = useState(true)
  const [selClassId,  setSelClassId]  = useState<string | null>(null) // null = Unassigned
  const [selModId,    setSelModId]    = useState<string | null>(null)
  const [pendingDocs, setPendingDocs] = useState<string[] | null>(null)
  const [libFilter,   setLibFilter]   = useState('')
  const [activeDrag,  setActiveDrag]  = useState<ActiveDrag | null>(null)
  const [crudError,   setCrudError]   = useState<{mod?: string; cls?: string}>({})
  const [renaming,    setRenaming]    = useState<{type: 'mod'|'cls'; id: string; value: string} | null>(null)
  const mutatingRef = useRef(false)  // serialise writes

  // ── Derived ───────────────────────────────────────────────────
  const classGroups    = board?.classes ?? []
  const allLibrary     = board?.library ?? []
  const selectedGroup  = classGroups.find(g => g.class_id === selClassId) ?? classGroups[0] ?? null
  const selectedModule = selectedGroup?.modules.find(m => m.id === selModId) ?? null
  const moduleDocs     = pendingDocs ?? selectedModule?.documents ?? []
  const filteredLib    = libFilter
    ? allLibrary.filter(f => f.toLowerCase().includes(libFilter.toLowerCase()))
    : allLibrary
  const moduleInSet    = new Set(moduleDocs)

  // ── Fetch ─────────────────────────────────────────────────────
  const fetchBoard = useCallback(async () => {
    try {
      const r = await fetch('/api/modules/board')
      if (r.ok) setBoard(await r.json() as BoardData)
    } catch (err) {
      console.error('[BOARD]', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchBoard() }, [fetchBoard])

  // Initialize class selection once board loads
  useEffect(() => {
    if (board && selectedGroup === null) {
      setSelClassId(board.classes[0]?.class_id ?? null)
    }
  }, [board]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Mutation helper ───────────────────────────────────────────
  const mutate = useCallback(async (
    url: string,
    method: string,
    body?: object,
    successMsg?: string,
  ) => {
    if (mutatingRef.current) return false
    mutatingRef.current = true
    try {
      await apiFetch(url, method, body)
      if (successMsg) onToast(successMsg)
      await fetchBoard()
      return true
    } catch (err) {
      onToast('Error', err instanceof Error ? err.message : String(err))
      await fetchBoard()
      return false
    } finally {
      mutatingRef.current = false
    }
  }, [fetchBoard, onToast])

  // ── Module-docs write (with optimistic update) ────────────────
  const writeDocs = useCallback(async (moduleId: string, docs: string[]) => {
    setPendingDocs(docs)
    if (mutatingRef.current) { setPendingDocs(null); return }
    mutatingRef.current = true
    try {
      await apiFetch(`/api/modules/${moduleId}/documents`, 'PUT', { filenames: docs })
    } catch (err) {
      onToast('Error', err instanceof Error ? err.message : String(err))
    } finally {
      mutatingRef.current = false
      await fetchBoard()
      setPendingDocs(null)
    }
  }, [fetchBoard, onToast])

  // ── dnd-kit sensors ──────────────────────────────────────────
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } })
  )

  function handleDragStart(event: DragStartEvent) {
    const d = event.active.data.current as { type: DragType; label?: string; filename?: string; moduleId?: string } | undefined
    if (!d) return
    const label = d.label ?? d.filename ?? ''
    setActiveDrag({ type: d.type, label })
  }

  async function handleDragEnd(event: DragEndEvent) {
    setActiveDrag(null)
    const { active, over } = event
    if (!over) return

    const aData = active.data.current as { type: DragType; moduleId?: string; filename?: string } | undefined
    const oData = over.data.current  as { type: string; classId?: string | null; filename?: string } | undefined
    if (!aData) return

    // ── Module dragged onto a class ──────────────────────────
    if (aData.type === 'module' && oData?.type === 'class-drop') {
      const moduleId = aData.moduleId!
      const classId  = oData.classId ?? null
      // Optimistic: move module in local board
      setBoard(prev => {
        if (!prev) return prev
        const allMods = prev.classes.flatMap(g => g.modules)
        const mod = allMods.find(m => m.id === moduleId)
        if (!mod) return prev
        const updated = { ...mod, class_id: classId }
        return {
          ...prev,
          classes: prev.classes.map(g => ({
            ...g,
            modules: g.class_id === classId
              ? [...g.modules.filter(m => m.id !== moduleId), updated]
              : g.modules.filter(m => m.id !== moduleId),
          })),
        }
      })
      // Find class name for toast
      const targetName = board?.classes.find(g => g.class_id === classId)?.name ?? 'Unassigned'
      await mutate(`/api/modules/${moduleId}/class`, 'POST', { class_id: classId },
        `Module moved to ${targetName}`)
    }

    // ── Library doc dropped into module docs ──────────────────
    else if (aData.type === 'lib-doc') {
      if (!selModId) return
      const filename = aData.filename!
      if (moduleInSet.has(filename)) return  // already present
      let newDocs = [...moduleDocs]
      // Insert at hovered mod-doc position, or append
      if (oData?.type === 'mod-doc' && oData.filename) {
        const idx = newDocs.indexOf(oData.filename)
        if (idx >= 0) newDocs.splice(idx, 0, filename)
        else newDocs.push(filename)
      } else {
        newDocs.push(filename)
      }
      // Optimistic update fires synchronously in the same render as
      // setActiveDrag(null) above — item appears in col 3 the instant
      // the overlay disappears, no snap-back frame.
      setPendingDocs(newDocs)
      await writeDocs(selModId, newDocs)
    }

    // ── Module doc reordered within col 3 ────────────────────
    else if (aData.type === 'mod-doc') {
      if (!selModId) return
      const activeFile = aData.filename!
      const overFile   = oData?.filename
      if (!overFile || activeFile === overFile) return
      const oldIdx = moduleDocs.indexOf(activeFile)
      const newIdx = moduleDocs.indexOf(overFile)
      if (oldIdx < 0 || newIdx < 0) return
      await writeDocs(selModId, arrayMove([...moduleDocs], oldIdx, newIdx))
    }
  }

  // ── Remove doc from module ────────────────────────────────────
  const handleRemoveDoc = useCallback((filename: string) => {
    if (!selModId) return
    writeDocs(selModId, moduleDocs.filter(f => f !== filename))
  }, [selModId, moduleDocs, writeDocs])

  // ── Class CRUD ────────────────────────────────────────────────
  async function handleCreateClass(name: string) {
    setCrudError(e => ({ ...e, cls: undefined }))
    try {
      await apiFetch('/api/classes', 'POST', { name })
      await fetchBoard()
      onToast('Class created', name)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setCrudError(e => ({ ...e, cls: msg }))
      throw err
    }
  }

  async function handleRenameClass() {
    if (!renaming || renaming.type !== 'cls') return
    const name = renaming.value.trim()
    if (!name) { setCrudError(e => ({ ...e, cls: 'Name cannot be empty' })); return }
    const ok = await mutate(`/api/classes/${renaming.id}`, 'PATCH', { name }, `Class renamed to "${name}"`)
    if (ok) setRenaming(null)
  }

  async function handleDeleteClass() {
    if (!selectedGroup || selectedGroup.class_id === null) return
    const ok = await mutate(
      `/api/classes/${selectedGroup.class_id}`, 'DELETE', undefined,
      `Class "${selectedGroup.name}" deleted — modules moved to Unassigned`
    )
    if (ok) setSelClassId(null)
  }

  // ── Module CRUD ───────────────────────────────────────────────
  async function handleCreateModule(name: string) {
    setCrudError(e => ({ ...e, mod: undefined }))
    try {
      await apiFetch('/api/modules', 'POST', { name, class_id: selClassId })
      await fetchBoard()
      onToast('Module created', name)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setCrudError(e => ({ ...e, mod: msg }))
      throw err
    }
  }

  async function handleRenameModule() {
    if (!renaming || renaming.type !== 'mod') return
    const name = renaming.value.trim()
    if (!name) { setCrudError(e => ({ ...e, mod: 'Name cannot be empty' })); return }
    const ok = await mutate(`/api/modules/${renaming.id}`, 'PATCH', { name }, `Module renamed to "${name}"`)
    if (ok) setRenaming(null)
  }

  async function handleDeleteModule() {
    if (!selModId) return
    const modName = selectedModule?.name ?? selModId
    const ok = await mutate(`/api/modules/${selModId}`, 'DELETE', undefined,
      `Module "${modName}" deleted — documents still in library`)
    if (ok) setSelModId(null)
  }

  // ── Col-3 needs to know if lib-doc is actively being dragged ──
  const isDraggingLibDoc = activeDrag?.type === 'lib-doc'

  // ── Render ────────────────────────────────────────────────────
  if (loading) return (
    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-ghost)' }}>
      loading…
    </div>
  )

  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '20px 24px 32px' }}>

        {/* Header */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: '-0.02em' }}>Organize materials</div>
          <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            drag documents into modules · drag modules onto a class to assign
          </div>
        </div>

        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
        >
          {/* ── 4-column board ── */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1.1fr 1.1fr',
            gap: 10,
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 12,
            background: 'var(--bg-sidebar)',
            padding: 10,
            marginBottom: 22,
          }}>

            {/* Col 1 — CLASSES */}
            <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <ColLabel>CLASSES</ColLabel>
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', gap: 5,
                minHeight: 340,
                background: 'var(--surface)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 9,
                padding: 6,
              }}>
                {classGroups.map(group => (
                  <ClassRow
                    key={group.class_id ?? 'unassigned'}
                    group={group}
                    isSelected={group.class_id === selClassId}
                    moduleCount={group.modules.length}
                    onClick={() => {
                      setSelClassId(group.class_id)
                      setSelModId(null)
                    }}
                  />
                ))}
                <EmptySlot text="drop a module here" />
              </div>
            </div>

            {/* Col 2 — MODULES */}
            <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <ColLabel>
                MODULES
                {selectedGroup && (
                  <span style={{ color: 'var(--text-faint)', marginLeft: 6 }}>
                    · {selectedGroup.name.toLowerCase()}
                  </span>
                )}
              </ColLabel>
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', gap: 5,
                minHeight: 340,
                background: 'var(--surface)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 9,
                padding: 6,
              }}>
                {(selectedGroup?.modules ?? []).map((mod, i) => (
                  <ModuleRow
                    key={mod.id}
                    module={mod}
                    index={i}
                    isSelected={mod.id === selModId}
                    onClick={() => setSelModId(mod.id)}
                  />
                ))}
                {(selectedGroup?.modules ?? []).length === 0 && (
                  <EmptySlot text="no modules yet — create one below" />
                )}
              </div>
            </div>

            {/* Col 3 — DOCUMENTS IN MODULE */}
            <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <ColLabel>
                {selectedModule
                  ? selectedModule.name.toUpperCase()
                  : 'DOCUMENTS'}
                <span style={{ color: 'var(--text-faint)', marginLeft: 'auto' }}>
                  {moduleDocs.length > 0 ? `${moduleDocs.length} docs` : ''}
                </span>
              </ColLabel>
              <SortableContext
                items={moduleDocs.map(f => `moddoc:${f}`)}
                strategy={verticalListSortingStrategy}
              >
                <ModuleDocsZone highlightDrop={isDraggingLibDoc}>
                  {!selectedModule ? (
                    <EmptySlot text="select a module to see its documents" />
                  ) : (
                    <>
                      {moduleDocs.map((f, i) => (
                        <ModDocRow
                          key={f}
                          filename={f}
                          index={i}
                          onRemove={handleRemoveDoc}
                        />
                      ))}
                      <EmptySlot text="drag documents here →" />
                    </>
                  )}
                </ModuleDocsZone>
              </SortableContext>
            </div>

            {/* Col 4 — LIBRARY */}
            <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <ColLabel>DOCUMENT LIBRARY</ColLabel>
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column',
                minHeight: 340,
                background: 'var(--surface)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 9,
                padding: 6,
              }}>
                <input
                  placeholder="Filter…"
                  value={libFilter}
                  onChange={e => setLibFilter(e.target.value)}
                  style={{
                    height: 28,
                    background: 'var(--bg-base)',
                    border: '1px solid var(--line)',
                    borderRadius: 6,
                    outline: 'none',
                    color: 'var(--text)',
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: 11,
                    padding: '0 9px',
                    marginBottom: 6,
                  }}
                />
                <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {filteredLib.length === 0 ? (
                    <div style={{
                      textAlign: 'center',
                      padding: '16px 8px',
                      fontFamily: '"JetBrains Mono", monospace',
                      fontSize: 10,
                      color: 'var(--text-ghost)',
                    }}>
                      {allLibrary.length === 0 ? 'no documents ingested yet' : 'no matches'}
                    </div>
                  ) : filteredLib.map(f => (
                    <LibDocRow
                      key={f}
                      filename={f}
                      inModule={moduleInSet.has(f)}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          <DragOverlay dropAnimation={null}>
            {activeDrag ? <DragGhost drag={activeDrag} /> : null}
          </DragOverlay>
        </DndContext>

        {/* ── CRUD section ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>

          {/* Modules CRUD */}
          <div>
            <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 11 }}>Modules</div>
            <CreateRow
              placeholder="New module name — e.g. Week 3 · Synaptic Plasticity"
              onSubmit={handleCreateModule}
              error={crudError.mod}
            />
            {selModId && selectedModule && (
              <div style={{ display: 'flex', gap: 8, marginTop: 9, alignItems: 'center' }}>
                {renaming?.type === 'mod' && renaming.id === selModId ? (
                  <>
                    <input
                      autoFocus
                      value={renaming.value}
                      onChange={e => setRenaming({ ...renaming, value: e.target.value })}
                      onKeyDown={e => { if (e.key === 'Enter') handleRenameModule(); if (e.key === 'Escape') setRenaming(null) }}
                      style={{
                        flex: 1, height: 34,
                        background: 'var(--surface)',
                        border: '1px solid var(--accent-ring)',
                        borderRadius: 8, outline: 'none',
                        color: 'var(--text)', fontFamily: 'inherit', fontSize: 12.5,
                        padding: '0 11px',
                      }}
                    />
                    <AccentBtn label="Save" onClick={handleRenameModule} />
                    <GhostBtn label="Cancel" onClick={() => setRenaming(null)} />
                  </>
                ) : (
                  <>
                    <div style={{
                      flex: 1, height: 34,
                      display: 'flex', alignItems: 'center',
                      padding: '0 11px',
                      background: 'var(--surface)',
                      border: '1px solid var(--line-input)',
                      borderRadius: 8,
                      fontSize: 12.5, color: 'var(--text-soft)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}>
                      {selectedModule.name}
                    </div>
                    <GhostBtn label="Rename" onClick={() =>
                      setRenaming({ type: 'mod', id: selModId, value: selectedModule.name })
                    } />
                    <GhostBtn label="Delete" danger onClick={handleDeleteModule} />
                  </>
                )}
              </div>
            )}
            <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 7 }}>
              Deleting a module only removes its document references — files and embeddings stay in the library.
            </div>
          </div>

          {/* Classes CRUD */}
          <div>
            <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 11 }}>Classes</div>
            <CreateRow
              placeholder="New class name — e.g. NSC 4310"
              onSubmit={handleCreateClass}
              error={crudError.cls}
            />
            {selectedGroup && selectedGroup.class_id !== null && (
              <div style={{ display: 'flex', gap: 8, marginTop: 9, alignItems: 'center' }}>
                {renaming?.type === 'cls' && renaming.id === selectedGroup.class_id ? (
                  <>
                    <input
                      autoFocus
                      value={renaming.value}
                      onChange={e => setRenaming({ ...renaming, value: e.target.value })}
                      onKeyDown={e => { if (e.key === 'Enter') handleRenameClass(); if (e.key === 'Escape') setRenaming(null) }}
                      style={{
                        flex: 1, height: 34,
                        background: 'var(--surface)',
                        border: '1px solid var(--accent-ring)',
                        borderRadius: 8, outline: 'none',
                        color: 'var(--text)', fontFamily: 'inherit', fontSize: 12.5,
                        padding: '0 11px',
                      }}
                    />
                    <AccentBtn label="Save" onClick={handleRenameClass} />
                    <GhostBtn label="Cancel" onClick={() => setRenaming(null)} />
                  </>
                ) : (
                  <>
                    <div style={{
                      flex: 1, height: 34,
                      display: 'flex', alignItems: 'center',
                      padding: '0 11px',
                      background: 'var(--surface)',
                      border: '1px solid var(--line-input)',
                      borderRadius: 8,
                      fontSize: 12.5, color: 'var(--text-soft)',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {selectedGroup.name}
                    </div>
                    <GhostBtn label="Rename" onClick={() =>
                      setRenaming({ type: 'cls', id: selectedGroup.class_id!, value: selectedGroup.name })
                    } />
                    <GhostBtn label="Delete" danger onClick={handleDeleteClass} />
                  </>
                )}
              </div>
            )}
            {selectedGroup?.class_id === null && (
              <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 16 }}>
                "Unassigned" is a built-in group — it cannot be renamed or deleted.
              </div>
            )}
            <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 7 }}>
              Deleting a class moves its modules to Unassigned — no modules or files are deleted.
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
