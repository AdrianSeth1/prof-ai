import { useEffect, useState } from 'react'
import type { HealthData } from '../types'

const POLL_INTERVAL = 30_000

const LOADING: HealthData = {
  backend: { status: 'degraded', info: 'connecting…' },
  ollama:  { status: 'degraded', info: '…' },
  gpu:     { status: 'degraded', info: '…' },
}

export function useHealthStatus(): HealthData {
  const [data, setData] = useState<HealthData>(LOADING)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const res = await fetch('/api/health', { signal: AbortSignal.timeout(4000) })
        if (!cancelled && res.ok) {
          const json = await res.json() as HealthData
          setData(json)
        }
      } catch {
        if (!cancelled) {
          setData(d => ({
            ...d,
            backend: { status: 'down', info: 'unreachable' },
          }))
        }
      }
    }

    poll()
    const t = setInterval(poll, POLL_INTERVAL)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [])

  return data
}
