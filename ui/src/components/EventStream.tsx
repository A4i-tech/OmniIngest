import { useEffect, useRef } from 'react'
import type { RunEvent } from '../types'

interface EventStreamProps {
  events: RunEvent[]
  running: boolean
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n)
}

function formatTimestamp(t: number): string {
  const ms = Math.max(0, t)
  const min = Math.floor(ms / 60000)
  const sec = Math.floor((ms % 60000) / 1000)
  const tenth = Math.floor((ms % 1000) / 100)
  return `${pad2(min)}:${pad2(sec)}.${tenth}`
}

const KIND = (text: string) => <span style={{ color: 'var(--c62)' }}>{text}</span>

function eventText(e: RunEvent) {
  switch (e.kind) {
    case 'pipeline_begin':
      return e.message ? (
        <>
          {KIND(e.kind)} · {e.message}
        </>
      ) : (
        KIND(e.kind)
      )
    case 'step_begin':
      return (
        <>
          {KIND(e.kind)} · {e.step ?? ''}
          {e.stage && <span style={{ color: 'var(--c40)' }}> [stage {e.stage}]</span>}
        </>
      )
    case 'step_progress':
      return (
        <>
          {KIND(e.kind)} · {e.step ?? ''} · {e.completed ?? 0} / {e.total ?? '?'}
          {e.message && <> · {e.message}</>}
        </>
      )
    case 'step_end': {
      const status = e.status ?? 'success'
      const statusColor = status === 'success' ? 'var(--cream)' : status === 'failure' ? 'var(--failed)' : 'var(--c50)'
      return (
        <>
          {KIND(e.kind)} · {e.step ?? ''} · <span style={{ color: statusColor }}>{status}</span>
          {e.error && <span style={{ color: 'var(--failed)' }}> · {e.error}</span>}
        </>
      )
    }
    case 'pipeline_end': {
      // This event carries the run vocabulary (succeeded/failed), not a StepStatus.
      const status = e.status ?? 'succeeded'
      const text = `${e.kind} · ${status}`
      return status !== 'succeeded' ? <span style={{ color: 'var(--failed)' }}>{text}</span> : <>{text}</>
    }
    default:
      return KIND(e.kind)
  }
}

export default function EventStream({ events, running }: EventStreamProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const nearBottomRef = useRef(true)

  useEffect(() => {
    const el = scrollRef.current
    if (el && nearBottomRef.current) el.scrollTop = el.scrollHeight
  }, [events.length])

  function onScroll() {
    const el = scrollRef.current
    if (!el) return
    nearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 40
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div
        style={{
          height: 44,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 18px',
          borderBottom: '1px solid var(--c18)',
        }}
      >
        <span className="label">EVENT STREAM</span>
        {running && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="dot pulse" style={{ width: 5, height: 5, background: 'var(--running)' }} />
            <span className="mono" style={{ fontSize: 9, letterSpacing: '.14em', color: 'var(--c62)' }}>
              LIVE
            </span>
          </span>
        )}
      </div>
      {events.length === 0 ? (
        <div className="empty" style={{ padding: '12px 18px' }}>
          Waiting for the first event…
        </div>
      ) : (
        <div ref={scrollRef} onScroll={onScroll} className="scroll" style={{ flex: 1, minHeight: 0, padding: '12px 18px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {events.map((e, i) => {
              const isNewest = i === events.length - 1
              return (
                <div
                  key={i}
                  className="mono"
                  style={{ display: 'flex', gap: 8, fontSize: 10, lineHeight: 1.5, color: isNewest ? 'var(--running)' : undefined }}
                >
                  <span style={{ color: 'var(--c50)', flexShrink: 0 }}>{formatTimestamp(e.t)}</span>
                  <span>{eventText(e)}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
