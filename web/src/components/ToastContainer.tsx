import type { Toast } from '../types'

interface Props {
  toasts: Toast[]
  onDismiss: (id: string) => void
}

export default function ToastContainer({ toasts, onDismiss }: Props) {
  if (toasts.length === 0) return null

  return (
    <div style={{
      position: 'fixed',
      bottom: 18,
      right: 18,
      display: 'flex',
      flexDirection: 'column',
      gap: 9,
      zIndex: 80,
      width: 320,
      pointerEvents: 'none',
    }}>
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  )
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  return (
    <div
      className="animate-toast-in"
      onClick={() => onDismiss(toast.id)}
      style={{
        border: '1px solid rgba(255,255,255,0.10)',
        borderRadius: 10,
        background: 'var(--surface-toast)',
        boxShadow: '0 12px 36px rgba(0,0,0,0.5)',
        padding: '12px 13px',
        display: 'flex',
        gap: 11,
        cursor: 'pointer',
        pointerEvents: 'all',
      }}
    >
      <span style={{
        width: 7, height: 7,
        borderRadius: '50%',
        background: 'var(--accent)',
        flexShrink: 0,
        marginTop: 5,
        boxShadow: '0 0 6px rgba(94,106,210,0.6)',
      }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>{toast.title}</div>
        {toast.desc && (
          <div style={{
            fontSize: 11.5,
            color: 'var(--text-muted)',
            marginTop: 2,
            fontFamily: '"JetBrains Mono", monospace',
          }}>{toast.desc}</div>
        )}
      </div>
    </div>
  )
}
