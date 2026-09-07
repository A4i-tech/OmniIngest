import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { Run, RunStatus, RunStepState, StepStatus } from '../types'
import RunList from '../components/RunList'
import Timeline from '../components/Timeline'
import EventStream from '../components/EventStream'

interface RunsProps {
  runs: Run[]
  activeId: string | null
  onSelect: (id: string) => void
  onCancel: (id: string) => void
  connected: boolean
  error: string | null
}

function colorOf(status: RunStatus | StepStatus): string {
  switch (status) {
    case 'queued':
    case 'pending':
      return 'var(--c40)'
    case 'running':
      return 'var(--running)'
    case 'succeeded':
    case 'success':
      return 'var(--cream)'
    case 'failed':
    case 'failure':
      return 'var(--failed)'
    case 'cancelled':
    case 'skipped':
      return 'var(--c50)'
    default:
      return 'var(--c40)'
  }
}

function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n)
}

function formatElapsedTenths(ms: number): string {
  const clamped = Math.max(0, ms)
  const min = Math.floor(clamped / 60000)
  const sec = Math.floor((clamped % 60000) / 1000)
  const tenth = Math.floor((clamped % 1000) / 100)
  return `${pad2(min)}:${pad2(sec)}.${tenth}`
}

/** The producer only stamps `total` on the step_end that knows the final item count. */
function lastItemTotal(run: Run): number | undefined {
  for (let i = run.events.length - 1; i >= 0; i--) {
    const e = run.events[i]
    if (e.kind === 'step_end' && e.total !== undefined) return e.total
  }
  return undefined
}

function findCurrentStep(run: Run): RunStepState | null {
  const running = run.steps.find((s) => s.status === 'running')
  if (running) return running
  const started = run.steps.filter((s) => s.start !== undefined)
  return started.length > 0 ? started[started.length - 1] : null
}

function downloadRun(run: Run) {
  const blob = new Blob([JSON.stringify(run, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${run.run_id}.json`
  a.click()
  URL.revokeObjectURL(url)
}

function StatPair({ label, value, valueColor }: { label: string; value: string; valueColor?: string }) {
  return (
    <div>
      <div className="label-sm">{label}</div>
      <div className="mono" style={{ fontSize: 12, marginTop: 4, color: valueColor }}>
        {value}
      </div>
    </div>
  )
}

function StatCell({ label, color, children }: { label: string; color?: string; children: ReactNode }) {
  return (
    <div style={{ background: 'var(--blue)', padding: '13px 14px' }}>
      <div className="label-sm">{label}</div>
      <div className="serif" style={{ fontSize: 27, lineHeight: 1, marginTop: 6, color }}>
        {children}
      </div>
    </div>
  )
}

function ConnectionBanner({ error, connected }: { error: string | null; connected: boolean }) {
  if (error) {
    return (
      <div style={{ borderBottom: '1px solid var(--failed)', padding: '10px 24px', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)' }}>
        {error}
      </div>
    )
  }
  if (!connected) {
    return (
      <div style={{ borderBottom: '1px solid var(--failed)', padding: '10px 24px', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--c62)' }}>
        Live updates disconnected. Reconnecting…
      </div>
    )
  }
  return null
}

function Empty({ message }: { message: string }) {
  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <span className="empty">{message}</span>
    </div>
  )
}

export default function Runs({ runs, activeId, onSelect, onCancel, connected, error }: RunsProps) {
  const activeRun = runs.find((r) => r.run_id === activeId) ?? null

  const [, setTick] = useState(0)
  useEffect(() => {
    if (!activeRun || activeRun.status !== 'running') return
    const id = setInterval(() => setTick((t) => t + 1), 1000)
    return () => clearInterval(id)
  }, [activeRun?.run_id, activeRun?.status])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, width: '100%' }}>
      <ConnectionBanner error={error} connected={connected} />
      {runs.length === 0 ? (
        <Empty message="No runs yet. Build a pipeline and press RUN." />
      ) : (
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          <div style={{ width: 300, flexShrink: 0, borderRight: '1px solid var(--c22)' }}>
            <RunList runs={runs} activeId={activeId} onSelect={onSelect} colorOf={colorOf} />
          </div>

          <div style={{ flex: 1, minWidth: 0, borderRight: '1px solid var(--c22)', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            {!activeRun ? (
              <Empty message="Select a run to view its details." />
            ) : (
              <>
                <div style={{ padding: '20px 24px 18px', borderBottom: '1px solid var(--c18)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <span className="serif" style={{ fontSize: 36 }}>
                      {activeRun.pipeline_id}
                    </span>
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 6,
                        border: `1px solid ${colorOf(activeRun.status)}`,
                        color: colorOf(activeRun.status),
                        padding: '4px 10px',
                        fontFamily: 'var(--mono)',
                        fontSize: 9,
                        letterSpacing: '.18em',
                      }}
                    >
                      <span className="dot" style={{ width: 5, height: 5, background: colorOf(activeRun.status) }} />
                      {activeRun.status.toUpperCase()}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: 34, marginTop: 14 }}>
                    <StatPair label="RUN ID" value={activeRun.run_id.slice(0, 13)} />
                    <StatPair label="TENANT" value={activeRun.tenant} />
                    <StatPair
                      label="ELAPSED"
                      value={formatElapsedTenths(
                        activeRun.status === 'running'
                          ? Date.now() - activeRun.started_at
                          : (activeRun.ended_at ?? activeRun.started_at) - activeRun.started_at,
                      )}
                    />
                    <StatPair label="ITEMS" value={lastItemTotal(activeRun) !== undefined ? String(lastItemTotal(activeRun)) : '—'} />
                    <StatPair label="OUTPUT" value={activeRun.output} valueColor="var(--jq)" />
                  </div>
                </div>

                <div className="scroll" style={{ flex: 1, minHeight: 0, padding: '18px 24px 24px' }}>
                  <Timeline steps={activeRun.steps} startedAt={activeRun.started_at} endedAt={activeRun.ended_at} colorOf={colorOf} />

                  {(() => {
                    const currentStep = findCurrentStep(activeRun)
                    if (!currentStep) return null
                    const finished = activeRun.ended_at !== undefined
                    const nowOffset = Date.now() - activeRun.started_at
                    const stepStart = currentStep.start ?? 0
                    const stepEnd = currentStep.status === 'running' ? nowOffset : (currentStep.end ?? nowOffset)
                    const stepEvents = activeRun.events.filter((e) => e.step === currentStep.agent)
                    const retries = stepEvents.filter((e) => e.message?.toLowerCase().includes('retry')).length
                    return (
                      <div style={{ margin: '22px 0 0', borderTop: '1px solid var(--c18)', paddingTop: 16 }}>
                        <div className="label" style={{ marginBottom: 10 }}>
                          {finished ? 'LAST STEP' : 'CURRENT STEP'} · {currentStep.agent}
                        </div>
                        <div
                          style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
                            gap: 1,
                            background: 'var(--c18)',
                            border: '1px solid var(--c18)',
                          }}
                        >
                          <StatCell label="COMPLETED">
                            {currentStep.completed ?? 0}
                            {currentStep.total !== undefined && (
                              <span className="serif" style={{ fontSize: 15, color: 'var(--c50)' }}>
                                {' '}
                                / {currentStep.total}
                              </span>
                            )}
                          </StatCell>
                          <StatCell label="EVENTS">{stepEvents.length}</StatCell>
                          <StatCell label="RETRIES" color={retries > 0 ? 'var(--failed)' : undefined}>
                            {retries}
                          </StatCell>
                          <StatCell label="ELAPSED">{Math.max(0, Math.round((stepEnd - stepStart) / 1000))}s</StatCell>
                        </div>
                      </div>
                    )
                  })()}
                </div>
              </>
            )}
          </div>

          <div style={{ width: 372, flexShrink: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            {!activeRun ? (
              <Empty message="Select a run to view its details." />
            ) : (
              <>
                <div style={{ flex: 1, minHeight: 0 }}>
                  <EventStream events={activeRun.events} running={activeRun.status === 'running'} />
                </div>
                <div style={{ borderTop: '1px solid var(--c18)', padding: '14px 18px', display: 'flex', gap: 8 }}>
                  <button className="btn" style={{ flex: 1, justifyContent: 'center' }} onClick={() => downloadRun(activeRun)}>
                    DOWNLOAD JSON
                  </button>
                  {(activeRun.status === 'running' || activeRun.status === 'queued') && (
                    <button className="btn btn-danger" style={{ flex: 1, justifyContent: 'center' }} onClick={() => onCancel(activeRun.run_id)}>
                      CANCEL RUN
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
