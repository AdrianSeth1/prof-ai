import { useCallback, useEffect, useRef, useState } from 'react'
import type { Screen, Mode, Toast, PaletteCommand } from './types'
import { useHealthStatus } from './hooks/useHealthStatus'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import CommandPalette from './components/CommandPalette'
import ToastContainer from './components/ToastContainer'
import Chat from './screens/Chat'
import LiveLecture from './screens/LiveLecture'
import AddMaterials from './screens/AddMaterials'
import Modules from './screens/Modules'
import GapsAnalysis from './screens/GapsAnalysis'

let toastSeq = 0

export default function App() {
  const [screen, setScreen] = useState<Screen>('chat')
  const [mode] = useState<Mode>('lecture')
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [paletteQuery, setPaletteQuery] = useState('')
  const [toasts, setToasts] = useState<Toast[]>([])
  const health = useHealthStatus()

  const goTo = useCallback((s: Screen) => {
    setScreen(s)
    setPaletteOpen(false)
  }, [])

  const openPalette = useCallback(() => {
    setPaletteQuery('')
    setPaletteOpen(true)
  }, [])

  const closePalette = useCallback(() => setPaletteOpen(false), [])

  const addToast = useCallback((title: string, desc?: string) => {
    const id = String(++toastSeq)
    setToasts(t => [...t, { id, title, desc }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4600)
  }, [])

  const dismissToast = useCallback((id: string) => {
    setToasts(t => t.filter(x => x.id !== id))
  }, [])

  const commands: PaletteCommand[] = [
    { label: 'Go to Chat',         hint: '1', group: 'Navigation', run: () => goTo('chat') },
    { label: 'Go to Live Lecture', hint: '2', group: 'Navigation', run: () => goTo('live') },
    { label: 'Add Materials',      hint: '3', group: 'Navigation', run: () => goTo('materials') },
    { label: 'Modules',            hint: '4', group: 'Navigation', run: () => goTo('modules') },
    { label: 'Gaps Analysis',      hint: '5', group: 'Navigation', run: () => goTo('gaps') },
    { label: 'Ask AI a question',  group: 'Lecture',   run: () => goTo('live') },
    { label: 'Re-run gap analysis',group: 'Actions',   run: () => goTo('gaps') },
    { label: 'Ingest a document…', group: 'Actions',   run: () => goTo('materials') },
    {
      label: 'Start new session',
      group: 'Actions',
      run: () => addToast("New session created", "Spring '26 · Lecture 13"),
    },
  ]

  // Global keyboard handler
  const paletteOpenRef = useRef(paletteOpen)
  paletteOpenRef.current = paletteOpen

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const k = e.key

      // Cmd/Ctrl-K: toggle palette
      if ((e.metaKey || e.ctrlKey) && k === 'k') {
        e.preventDefault()
        if (paletteOpenRef.current) {
          setPaletteOpen(false)
        } else {
          setPaletteQuery('')
          setPaletteOpen(true)
        }
        return
      }

      // Esc: close palette
      if (k === 'Escape' && paletteOpenRef.current) {
        setPaletteOpen(false)
        return
      }

      // Number keys 1-5: navigate screens (when no input focused)
      const tag = (e.target as HTMLElement | null)?.tagName ?? ''
      if (!paletteOpenRef.current && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        const map: Record<string, Screen> = {
          '1': 'chat', '2': 'live', '3': 'materials', '4': 'modules', '5': 'gaps',
        }
        if (map[k]) goTo(map[k])
      }
    }

    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [goTo])

  const screenTitles: Record<Screen, string> = {
    chat:      'Chat',
    live:      'Live Lecture',
    materials: 'Add Materials',
    modules:   'Modules',
    gaps:      'Gaps Analysis',
  }

  const screenSubs: Record<Screen, string> = {
    chat:      'rag · literature',
    live:      'recording · lecture-12',
    materials: 'vector store',
    modules:   'nsc-4310',
    gaps:      'batch · lecture-12',
  }

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100%', overflow: 'hidden', background: 'var(--bg-base)' }}>
      <Sidebar
        screen={screen}
        mode={mode}
        health={health}
        onNavigate={goTo}
        onOpenPalette={openPalette}
      />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <TopBar
          title={screenTitles[screen]}
          sub={screenSubs[screen]}
          mode={mode}
          onOpenPalette={openPalette}
        />

        <main style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
          {screen === 'chat'      && <Chat />}
          {screen === 'live'      && <LiveLecture onToast={addToast} />}
          {screen === 'materials' && <AddMaterials onToast={addToast} />}
          {screen === 'modules'   && <Modules onToast={addToast} />}
          {screen === 'gaps'      && <GapsAnalysis onToast={addToast} />}
        </main>
      </div>

      {paletteOpen && (
        <CommandPalette
          query={paletteQuery}
          onQueryChange={setPaletteQuery}
          commands={commands}
          onClose={closePalette}
        />
      )}

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  )
}
