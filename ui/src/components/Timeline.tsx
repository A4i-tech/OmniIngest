import { useEffect, useState } from 'react'
import type { RunStepState, StepStatus } from '../types'

interface TimelineProps {
  steps: RunStepState[]
  startedAt: number
  endedAt?: number
  colorOf: (status: StepStatus) => string
}

function stepRow(step: RunStepState, durationMs: number, nowOffset: number, colorOf: (s: StepStatus) => string, key: number) {
  const pending = step.status === 'pending'
  const running = step.status === 'running'
  const hasBar = step.start !== undefined
  const startOffset = step.start ?? 0
  const endOffset = running ? nowOffset : (step.end ?? startOffset)
  const left = hasBar ? (startOffset / durationMs) * 100 : 0
  const width = hasBar ? Math.max(0, ((endOffset - startOffset) / durationMs) * 100) : 0

  let durationLabel = '—'
  if (!pending) durationLabel = `${Math.max(0, Math.round((endOffset - startOffset) / 1000))}s`

  return (
    <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
      <div style={{ width: 132, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
        <span
          className="sq"
          style={pending ? { border: `1px solid ${colorOf('pending')}` } : { background: colorOf(step.status) }}
        />
        <span className="mono" style={{ fontSize: 11, color: pending ? 'var(--c40)' : undefined }}>
          {step.agent}
        </span>
      </div>
      <div style={{ flex: 1, height: 20, background: 'var(--c10)', position: 'relative' }}>
        {hasBar && (
          <div
            style={{
              position: 'absolute',
              left: `${left}%`,
              width: `${width}%`,
              top: 0,
              bottom: 0,
              background: colorOf(step.status),
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
            }}
          >
            {running && width >= 15 && step.total !== undefined && (
              <span className="mono" style={{ fontSize: 9, color: 'var(--blue)', fontWeight: 600, paddingRight: 7 }}>
                {step.completed ?? 0} / {step.total}
              </span>
            )}
          </div>
        )}
      </div>
      <div
        className="mono"
        style={{ width: 64, textAlign: 'right', fontSize: 10, color: pending ? 'var(--c40)' : running ? 'var(--running)' : 'var(--c62)' }}
      >
        {durationLabel}
      </div>
    </div>
  )
}

export default function Timeline({ steps, startedAt, endedAt, colorOf }: TimelineProps) {
  const [, setTick] = useState(0)
  useEffect(() => {
    if (endedAt !== undefined) return
    const id = setInterval(() => setTick((t) => t + 1), 250)
    return () => clearInterval(id)
  }, [endedAt])

  const nowOffset = Date.now() - startedAt
  const durationMs = Math.max(1, endedAt !== undefined ? endedAt - startedAt : nowOffset)
  const totalSeconds = Math.max(1, Math.round(durationMs / 1000))

  const groups: RunStepState[][] = []
  for (const step of steps) {
    const prevGroup = groups[groups.length - 1]
    if (step.stage && prevGroup && prevGroup[0].stage === step.stage) prevGroup.push(step)
    else groups.push([step])
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
        <span className="label">STEP TIMELINE</span>
        <span className="label">0s — {totalSeconds}s</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
        {groups.map((group, gi) =>
          group.length > 1 ? (
            <div
              key={gi}
              style={{ borderLeft: '1px dashed var(--c40)', paddingLeft: 10, marginLeft: -10, display: 'flex', flexDirection: 'column', gap: 9 }}
            >
              {group.map((step, si) => stepRow(step, durationMs, nowOffset, colorOf, si))}
            </div>
          ) : (
            stepRow(group[0], durationMs, nowOffset, colorOf, gi)
          ),
        )}
      </div>
    </div>
  )
}
