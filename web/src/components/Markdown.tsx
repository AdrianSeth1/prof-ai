import type { CSSProperties } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'

// Renders assistant Markdown to match the §1 design tokens — chat-scale
// (headings modest, not document-sized), tight list/paragraph spacing.
const components: Components = {
  h1: ({ children }) => (
    <h1 style={{ fontSize: 15, fontWeight: 600, letterSpacing: '-0.02em', margin: '14px 0 6px', lineHeight: 1.3, color: 'var(--text)' }}>{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-0.02em', margin: '12px 0 5px', lineHeight: 1.3, color: 'var(--text)' }}>{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 style={{ fontSize: 13.5, fontWeight: 600, letterSpacing: '-0.01em', margin: '10px 0 4px', lineHeight: 1.3, color: 'var(--text)' }}>{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 style={{ fontSize: 13, fontWeight: 600, margin: '8px 0 4px', lineHeight: 1.3, color: 'var(--text-soft)' }}>{children}</h4>
  ),
  h5: ({ children }) => (
    <h5 style={{ fontSize: 12.5, fontWeight: 600, margin: '8px 0 4px', lineHeight: 1.3, color: 'var(--text-soft)' }}>{children}</h5>
  ),
  h6: ({ children }) => (
    <h6 style={{ fontSize: 12.5, fontWeight: 600, margin: '8px 0 4px', lineHeight: 1.3, color: 'var(--text-muted)' }}>{children}</h6>
  ),
  p: ({ children }) => (
    <p style={{ margin: '0 0 8px', lineHeight: 1.68 }}>{children}</p>
  ),
  ul: ({ children }) => (
    <ul style={{ margin: '0 0 8px', paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 2 }}>{children}</ul>
  ),
  ol: ({ children }) => (
    <ol style={{ margin: '0 0 8px', paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 2 }}>{children}</ol>
  ),
  li: ({ children }) => (
    <li style={{ lineHeight: 1.6 }}>{children}</li>
  ),
  blockquote: ({ children }) => (
    <blockquote style={{
      margin: '4px 0 8px',
      padding: '1px 0 1px 12px',
      borderLeft: '2px solid var(--accent)',
      color: 'var(--text-muted)',
    }}>{children}</blockquote>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{ color: 'var(--accent-light)', textDecoration: 'underline', textUnderlineOffset: 2 }}
    >
      {children}
    </a>
  ),
  hr: () => (
    <hr style={{ border: 'none', borderTop: '1px solid var(--line)', margin: '10px 0' }} />
  ),
  strong: ({ children }) => (
    <strong style={{ color: 'var(--text)', fontWeight: 600 }}>{children}</strong>
  ),
  em: ({ children }) => (
    <em style={{ color: 'var(--text-soft)' }}>{children}</em>
  ),
  table: ({ children }) => (
    <div style={{ overflowX: 'auto', margin: '6px 0 10px' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 13 }}>{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead>{children}</thead>,
  th: ({ children }) => (
    <th style={{
      border: '1px solid var(--line)', padding: '5px 9px', textAlign: 'left',
      color: 'var(--text-soft)', fontWeight: 600, background: 'var(--surface-2)',
    }}>{children}</th>
  ),
  td: ({ children }) => (
    <td style={{ border: '1px solid var(--line)', padding: '5px 9px' }}>{children}</td>
  ),
  pre: ({ children }) => (
    <pre style={{
      background: 'var(--surface-pop)',
      border: '1px solid var(--line)',
      borderRadius: 8,
      padding: '10px 12px',
      overflowX: 'auto',
      margin: '6px 0 10px',
      lineHeight: 1.55,
    }}>{children}</pre>
  ),
  code(props) {
    const { className, children, ...rest } = props
    const text = String(children)
    // react-markdown v9+ no longer passes an `inline` flag. A fenced block always
    // carries a language- class when a language is given, and otherwise usually
    // spans multiple lines — that combination is enough to tell it apart from a
    // short inline `code` span without over-engineering a parent-node lookup.
    const isBlock = /language-/.test(className || '') || text.includes('\n')
    if (isBlock) {
      return (
        <code className={className} style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: 12.5, color: 'var(--text-body)' }} {...rest}>
          {children}
        </code>
      )
    }
    return (
      <code style={{
        background: 'var(--surface-pop)', color: 'var(--text-soft)',
        padding: '1px 5px', borderRadius: 4,
        fontFamily: '"JetBrains Mono", monospace', fontSize: '0.9em',
      }} {...rest}>
        {children}
      </code>
    )
  },
}

export default function Markdown({ text, style }: { text: string; style?: CSSProperties }) {
  return (
    <div style={{ fontSize: 14, color: 'var(--text-body)', lineHeight: 1.68, ...style }}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  )
}
