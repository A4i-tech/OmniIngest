import { useEffect, useState } from 'react'
import type { Run, RunStatus } from '../types'

interface RunListProps {
  runs: Run[]
  activeId: string | null
  onSelect: (id: string) => void
  colorOf: (status: RunStatus) => string
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n)
}

function formatMMSS(ms: number): string {
  const clamped = Math.max(0, ms)
  const min = Math.floor(clamped / 60000)
  const sec = Math.floor((clamped % 60000) / 1000)
  return `${pad2(min)}:${pad2(sec)}`
}

/** The producer only stamps `total` on the step_end that knows the final item count. */
function lastItemTotal(run: Run): number | undefined {
  for (let i = run.events.length - 1; i >= 0; i--) {
    const e = run.events[i]
    if (e.kind === 'step_end' && e.total !== undefined) return e.total
  }
  return undefined
}

const basename = (uri: string) => uri.split(/[/\\]/).pop() || uri

function Summary({ run }: { run: Run }) {
  if (run.status === 'running' || run.status === 'queued') return <>{basename(run.source)}</>
  if (run.status === 'succeeded') {
    const n = lastItemTotal(run)
    return <>{n !== undefined ? `${n} items · succeeded` : basename(run.source)}</>
  }
  if (run.status === 'failed') {
    const step = run.steps.find((s) => s.status === 'failure')
    return <span style={{ color: 'var(--failed)' }}>failed at {step ? step.agent : run.source}</span>
  }
  return <>cancelled</>
}

export default function RunList({ runs, activeId, onSelect, colorOf }: RunListProps) {
  const [, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px 18px 12px' }}>
        <span className="label">RUNS</span>
        <span className="label">{runs.length} TOTAL</span>
      </div>
      <div className="scroll" style={{ flex: 1, minHeight: 0 }}>
        {runs.map((run) => {
          const active = run.run_id === activeId
          const running = run.status === 'running'
          const elapsedMs = running ? Date.now() - run.started_at : run.ended_at !== undefined ? run.ended_at - run.started_at : 0
          const completed = run.steps.filter((s) => s.status === 'success' || s.status === 'failure' || s.status === 'skipped').length
          const fraction = run.steps.length > 0 ? completed / run.steps.length : 0
          return (
            <div
              key={run.run_id}
              onClick={() => onSelect(run.run_id)}
              style={{
                padding: '11px 16px 12px',
                borderLeft: `2px solid ${active ? colorOf(run.status) : 'transparent'}`,
                borderBottom: active ? 'none' : '1px solid var(--c18)',
                background: active ? 'var(--hover)' : 'transparent',
                cursor: 'pointer',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="dot" style={{ background: colorOf(run.status) }} />
                <span className="mono" style={{ fontSize: 11, flex: 1, minWidth: 0 }}>
                  {run.pipeline_id}
                </span>
                <span className="mono" style={{ fontSize: 10, color: running ? 'var(--running)' : 'var(--c62)' }}>
                  {formatMMSS(elapsedMs)}
                </span>
              </div>
              <div
                className="mono"
                style={{ fontSize: 9, color: 'var(--c50)', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              >
                {run.run_id.slice(0, 8)} · <Summary run={run} />
              </div>
              {running && (
                <div style={{ height: 3, background: 'var(--c10)', marginTop: 6 }}>
                  <div style={{ height: '100%', width: `${fraction * 100}%`, background: 'var(--running)' }} />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
