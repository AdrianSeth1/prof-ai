import { useCallback, useEffect, useRef, useState } from 'react'

// ── Local types ───────────────────────────────────────────────────

type FileStatus = 'queued' | 'processing' | 'ocr' | 'transcribing' | 'done' | 'skipped' | 'error'

interface IngestItem {
  id: string
  name: string
  isAudio: boolean
  status: FileStatus
  message: string
  chunks: number
  skipped: boolean
}

interface Material {
  filename: string
  file_type: string
  content_type: string
  chunks: number
}

interface Props {
  onToast: (title: string, desc?: string) => void
}

// ── Constants ─────────────────────────────────────────────────────

const DOC_EXTS   = new Set(['.pdf', '.txt', '.docx', '.pptx'])
const AUDIO_EXTS = new Set(['.mp3', '.wav', '.m4a', '.flac'])
const ACCEPT     = '.pdf,.docx,.pptx,.txt,.mp3,.wav,.m4a,.flac'

// ── Helpers ───────────────────────────────────────────────────────

function fileExt(name: string): string {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

function uid(): string { return Math.random().toString(36).slice(2, 10) }

// ── Sub-components ────────────────────────────────────────────────

function FileDocIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none"
      stroke="var(--text-muted)" strokeWidth="1.4" style={{ flexShrink: 0 }}>
      <path d="M4 2h5l3 3v9H4z"/>
      <path d="M9 2v3h3"/>
    </svg>
  )
}

function FileAudioIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none"
      stroke="var(--text-muted)" strokeWidth="1.4" style={{ flexShrink: 0 }}>
      <rect x="4" y="2" width="6" height="12" rx="1"/>
      <path d="M6.5 6.5h3M6.5 9h3"/>
    </svg>
  )
}

function IngestRow({ item }: { item: IngestItem }) {
  const isActive = item.status === 'queued' || item.status === 'processing'
    || item.status === 'ocr' || item.status === 'transcribing'
  const isOcr    = item.status === 'ocr'

  const barColor: string = isOcr ? 'var(--await)' : 'var(--accent)'

  let statusLabel: string
  switch (item.status) {
    case 'queued':       statusLabel = 'queued'; break
    case 'processing':   statusLabel = 'processing…'; break
    case 'ocr':          statusLabel = 'OCR fallback · tesseract'; break
    case 'transcribing': statusLabel = 'transcribing · whisper'; break
    case 'done':         statusLabel = `${item.chunks} chunk${item.chunks !== 1 ? 's' : ''} stored`; break
    case 'skipped':      statusLabel = 'skipped (unchanged)'; break
    case 'error':        statusLabel = item.message || 'error'; break
    default:             statusLabel = ''
  }

  const statusColor: string =
    item.status === 'error'   ? 'var(--rec-text)'   :
    item.status === 'skipped' ? 'var(--text-muted)'  :
    item.status === 'done'    ? 'var(--ok)'           :
    isOcr                     ? 'var(--await)'        :
    'var(--accent-light)'

  return (
    <div style={{
      border: '1px solid var(--line)',
      borderRadius: 9,
      background: 'var(--surface)',
      padding: '12px 14px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 12.5,
          flex: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          color: 'var(--text)',
        }}>
          {item.name}
        </span>
        <span style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 11,
          color: statusColor,
          flexShrink: 0,
          maxWidth: 240,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {isOcr && (
            <span className="animate-spin-slow" style={{
              display: 'inline-block',
              width: 11,
              height: 11,
              border: '2px solid rgba(233,162,59,0.25)',
              borderTopColor: 'var(--await)',
              borderRadius: '50%',
              flexShrink: 0,
            }} />
          )}
          {statusLabel}
        </span>
      </div>

      {/* Progress bar — visible while active or just completed */}
      {(isActive || item.status === 'done') && (
        <div style={{
          height: 4,
          borderRadius: 3,
          background: 'rgba(255,255,255,0.07)',
          marginTop: 9,
          overflow: 'hidden',
          position: 'relative',
        }}>
          {item.status === 'done' ? (
            <div style={{ width: '100%', height: '100%', background: 'var(--ok)', borderRadius: 3 }} />
          ) : (
            <div className="animate-ingest-bar" style={{ background: barColor }} />
          )}
        </div>
      )}
    </div>
  )
}

function MaterialRow({ m, last }: { m: Material; last: boolean }) {
  const isAudio = m.content_type === 'lecture_transcript'

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '11px 14px',
      borderBottom: last ? 'none' : '1px solid rgba(255,255,255,0.06)',
    }}>
      {isAudio ? <FileAudioIcon /> : <FileDocIcon />}

      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 12.5,
        flex: 1,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>
        {m.filename}
      </span>

      {/* content-type badge */}
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 10,
        padding: '1px 6px',
        border: `1px solid ${isAudio ? 'rgba(94,106,210,0.28)' : 'rgba(255,255,255,0.10)'}`,
        borderRadius: 4,
        color: isAudio ? 'var(--accent-light)' : 'var(--text-faint)',
        background: isAudio ? 'var(--accent-08)' : 'transparent',
        flexShrink: 0,
      }}>
        {isAudio ? 'transcript' : 'source doc'}
      </span>

      {/* chunk count */}
      <span style={{
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 10.5,
        color: 'var(--text-faint)',
        flexShrink: 0,
        minWidth: 64,
        textAlign: 'right',
      }}>
        {m.chunks} chunk{m.chunks !== 1 ? 's' : ''}
      </span>

      {/* status */}
      <span style={{
        display: 'flex',
        alignItems: 'center',
        gap: 5,
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 10.5,
        color: 'var(--ok)',
        flexShrink: 0,
      }}>
        <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--ok)', flexShrink: 0 }} />
        indexed
      </span>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────

export default function AddMaterials({ onToast }: Props) {
  const [items,    setItems]    = useState<IngestItem[]>([])
  const [materials, setMaterials] = useState<Material[]>([])
  const [mLoading, setMLoading] = useState(true)
  const [dragOver, setDragOver] = useState(false)
  const [rejected, setRejected] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const busyRef      = useRef(false)  // prevent concurrent uploads

  // ── Fetch materials list ─────────────────────────────────────────
  const fetchMaterials = useCallback(async () => {
    try {
      const r = await fetch('/api/materials')
      if (r.ok) {
        const data = await r.json()
        setMaterials(data.materials ?? [])
      }
    } catch (err) {
      console.error('[MATERIALS]', err)
    } finally {
      setMLoading(false)
    }
  }, [])

  useEffect(() => { fetchMaterials() }, [fetchMaterials])

  // ── Upload + stream ──────────────────────────────────────────────
  const uploadFiles = useCallback(async (fileList: File[]) => {
    if (busyRef.current) return

    // Partition into valid / rejected
    const valid: File[]    = []
    const bad:   string[]  = []
    for (const f of fileList) {
      const e = fileExt(f.name)
      if (DOC_EXTS.has(e) || AUDIO_EXTS.has(e)) valid.push(f)
      else bad.push(f.name)
    }
    if (bad.length) setRejected(bad)
    if (!valid.length) return

    // Add rows to the list immediately
    const newItems: IngestItem[] = valid.map(f => ({
      id:      uid(),
      name:    f.name,
      isAudio: AUDIO_EXTS.has(fileExt(f.name)),
      status:  'queued',
      message: '',
      chunks:  0,
      skipped: false,
    }))

    // Stable references to the IDs we just added, keyed by filename
    // (filename is unique within a single batch upload)
    const idByName: Record<string, string> = {}
    for (const it of newItems) idByName[it.name] = it.id

    setItems(prev => [...prev, ...newItems])

    const form = new FormData()
    for (const f of valid) form.append('files', f)

    busyRef.current = true
    try {
      const resp = await fetch('/api/ingest', { method: 'POST', body: form })
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf           = ''
      let totalChunks   = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''

        for (const raw of lines) {
          const line = raw.trim()
          if (!line) continue
          let parsed: Record<string, unknown>
          try { parsed = JSON.parse(line) } catch { continue }

          const file = parsed.file as string | undefined

          const patchItem = (patch: Partial<IngestItem>) => {
            if (!file) return
            setItems(prev => prev.map(it =>
              it.name === file ? { ...it, ...patch } : it
            ))
          }

          if (parsed.type === 'progress') {
            const msg = (parsed.message as string) ?? ''
            let status: FileStatus | undefined
            if (msg.startsWith('Transcribing')) status = 'transcribing'
            patchItem({ message: msg, ...(status ? { status } : {}) })
            // Transition from queued → processing on first progress event
            setItems(prev => prev.map(it => {
              if (it.name !== file) return it
              if (it.status === 'queued') return { ...it, status: 'processing', message: msg }
              if (status) return { ...it, status, message: msg }
              return { ...it, message: msg }
            }))
          } else if (parsed.type === 'ocr_started') {
            patchItem({ status: 'ocr', message: (parsed.message as string) ?? '' })
          } else if (parsed.type === 'file_done') {
            const chunks  = (parsed.chunks  as number)  ?? 0
            const skipped = (parsed.skipped as boolean) ?? false
            totalChunks  += chunks
            patchItem({ status: skipped ? 'skipped' : 'done', chunks, skipped })
          } else if (parsed.type === 'file_error') {
            patchItem({ status: 'error', message: (parsed.message as string) ?? 'Unknown error' })
          } else if (parsed.type === 'done') {
            totalChunks = (parsed.total_chunks as number) ?? totalChunks
          }
        }
      }

      await fetchMaterials()
      const desc = totalChunks > 0
        ? `${totalChunks} new chunk${totalChunks !== 1 ? 's' : ''} added to vector store`
        : 'All files skipped (unchanged) or already indexed'
      onToast('Ingest complete', desc)

    } catch (err) {
      console.error('[INGEST]', err)
      // Mark any still-queued or processing items as errored
      setItems(prev => prev.map(it =>
        (it.status === 'queued' || it.status === 'processing')
          ? { ...it, status: 'error', message: 'Upload failed' }
          : it
      ))
      onToast('Ingest failed', err instanceof Error ? err.message : String(err))
    } finally {
      busyRef.current = false
    }
  }, [fetchMaterials, onToast])

  // ── Drag-and-drop handlers ───────────────────────────────────────
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => setDragOver(false), [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    uploadFiles(Array.from(e.dataTransfer.files))
  }, [uploadFiles])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      uploadFiles(Array.from(e.target.files))
      e.target.value = ''  // reset so same file can trigger again
    }
  }, [uploadFiles])

  const activeCount = items.filter(i =>
    i.status === 'queued' || i.status === 'processing'
    || i.status === 'ocr'  || i.status === 'transcribing'
  ).length

  // ── Render ──────────────────────────────────────────────────────
  return (
    <div style={{ height: '100%', overflowY: 'auto' }}>
      <div style={{ maxWidth: 880, margin: '0 auto', padding: 24 }}>

        {/* ── Dropzone ── */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: `1.5px dashed ${dragOver ? 'rgba(94,106,210,0.4)' : 'rgba(255,255,255,0.14)'}`,
            borderRadius: 12,
            padding: 34,
            textAlign: 'center',
            background: dragOver ? '#0e0f13' : 'var(--surface)',
            cursor: 'pointer',
            transition: 'border-color 140ms ease, background 140ms ease',
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={ACCEPT}
            onChange={handleChange}
            style={{ display: 'none' }}
          />
          <div style={{
            width: 42, height: 42, borderRadius: 10,
            background: 'var(--accent-12)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 12px',
          }}>
            <svg width="20" height="20" viewBox="0 0 16 16" fill="none"
              stroke="var(--accent-light)" strokeWidth="1.4"
              strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 11V3.5M5 6l3-3 3 3"/>
              <path d="M3 11.5v1A1.5 1.5 0 004.5 14h7a1.5 1.5 0 001.5-1.5v-1"/>
            </svg>
          </div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>Drop files or click to browse</div>
          <div style={{ fontSize: 12.5, color: 'var(--text-ghost)', marginTop: 4 }}>
            PDF · Word · PowerPoint · audio (mp3, wav, m4a, flac) · OCR fallback for scans
          </div>
        </div>

        {/* ── Rejected types banner ── */}
        {rejected.length > 0 && (
          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            marginTop: 10,
            padding: '9px 12px',
            background: 'var(--rec-08)',
            border: '1px solid var(--rec-line)',
            borderRadius: 8,
          }}>
            <span style={{ fontSize: 12.5, color: 'var(--rec-text)', flex: 1, lineHeight: 1.5 }}>
              Unsupported file type: {rejected.join(', ')} — accepted: PDF, Word, PowerPoint, txt, mp3, wav, m4a, flac
            </span>
            <button
              onClick={() => setRejected([])}
              style={{
                background: 'none', border: 'none',
                color: 'var(--text-faint)', cursor: 'pointer',
                fontSize: 13, padding: 0, lineHeight: 1, flexShrink: 0,
              }}
            >✕</button>
          </div>
        )}

        {/* ── Active ingests ── */}
        {items.length > 0 && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '22px 0 11px' }}>
              <span style={{ fontSize: 12.5, fontWeight: 600 }}>Ingesting</span>
              {activeCount > 0 && (
                <span style={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: 10.5,
                  color: 'var(--text-faint)',
                }}>
                  {activeCount} active
                </span>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {items.map(item => <IngestRow key={item.id} item={item} />)}
            </div>
          </>
        )}

        {/* ── Ingested materials list ── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '24px 0 11px' }}>
          <span style={{ fontSize: 12.5, fontWeight: 600 }}>Ingested documents</span>
          <span style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 10.5,
            color: 'var(--text-faint)',
          }}>
            {mLoading ? '…' : `${materials.length} in vector store`}
          </span>
        </div>

        {!mLoading && materials.length === 0 ? (
          <div style={{
            padding: '22px 14px',
            textAlign: 'center',
            color: 'var(--text-ghost)',
            fontSize: 12.5,
            fontFamily: '"JetBrains Mono", monospace',
            border: '1px dashed var(--line)',
            borderRadius: 9,
          }}>
            no documents indexed yet — drop files above to get started
          </div>
        ) : (
          <div style={{
            border: '1px solid var(--line)',
            borderRadius: 9,
            background: 'var(--surface)',
            overflow: 'hidden',
          }}>
            {materials.map((m, i) => (
              <MaterialRow key={m.filename} m={m} last={i === materials.length - 1} />
            ))}
          </div>
        )}

      </div>
    </div>
  )
}
