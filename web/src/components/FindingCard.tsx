export type FindingStatus = 'covered' | 'partial' | 'uncovered'

export interface Finding {
  topic: string
  status: FindingStatus
  note: string
  evidence?: string
  source: string
}

// uncovered = amber (the actionable headline signal)
// partial   = neutral muted
// covered   = green, de-emphasized (the exception)
const STATUS_DOT: Record<FindingStatus, string> = {
  uncovered: 'var(--await)',
  partial:   'rgba(255,255,255,0.30)',
  covered:   'var(--ok)',
}

const STATUS_BORDER: Record<FindingStatus, string> = {
  uncovered: 'rgba(233,162,59,0.25)',
  partial:   'rgba(255,255,255,0.07)',
  covered:   'rgba(63,185,80,0.14)',
}

const STATUS_BADGE: Record<FindingStatus, string> = {
  uncovered: 'var(--await)',
  partial:   'rgba(255,255,255,0.35)',
  covered:   'var(--ok)',
}

export default function FindingCard({ finding }: { finding: Finding }) {
  const status = finding.status in STATUS_DOT ? finding.status : 'uncovered' as FindingStatus
  const dot    = STATUS_DOT[status]
  const border = STATUS_BORDER[status]
  const badge  = STATUS_BADGE[status]

  return (
    <div style={{
      border: `1px solid ${border}`,
      borderRadius: 10,
      background: 'var(--surface)',
      padding: '10px 13px',
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
    }}>
      {/* Topic row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: dot, flexShrink: 0,
        }} />
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', flex: 1, lineHeight: 1.4 }}>
          {finding.topic}
        </span>
        <span style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: 9.5,
          color: badge,
          opacity: status === 'covered' ? 0.65 : 0.9,
          flexShrink: 0,
        }}>
          {finding.status}
        </span>
      </div>

      {/* Note */}
      {finding.note && (
        <div style={{
          fontSize: 12.5, color: 'var(--text-body)', lineHeight: 1.55, paddingLeft: 15,
        }}>
          {finding.note}
        </div>
      )}

      {/* Evidence quote (only for covered/partial with actual evidence) */}
      {finding.evidence && (
        <div style={{ paddingLeft: 15, marginTop: 1 }}>
          <span style={{
            display: 'block',
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 10.5,
            color: 'var(--text-faint)',
            fontStyle: 'italic',
            lineHeight: 1.5,
            borderLeft: `2px solid ${dot}`,
            paddingLeft: 7,
            opacity: 0.85,
          }}>
            "{finding.evidence}"
          </span>
        </div>
      )}

      {/* Source */}
      {finding.source && (
        <div style={{ paddingLeft: 15, marginTop: finding.evidence ? 2 : 0 }}>
          <span style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: 10, color: 'var(--text-faint)',
          }}>
            {finding.source}
          </span>
        </div>
      )}
    </div>
  )
}
