import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'
import { ApiError } from './api'
import { applyEvent } from './runState'
import type { AgentSpec, Pipeline, Run, RunRequest, SavedPipeline } from './types'
import { emptyPipeline, fromYaml, toYaml } from './yaml'
import Builder from './views/Builder'
import Library from './views/Library'
import Runs from './views/Runs'

type View = 'builder' | 'runs' | 'library'

const describe = (e: unknown) =>
  e instanceof ApiError ? `${e.message} ${e.hint}` : e instanceof Error ? e.message : 'Something went wrong. Check the server log.'

export default function App() {
  const [view, setView] = useState<View>('builder')
  const [agents, setAgents] = useState<AgentSpec[]>([])
  const [saved, setSaved] = useState<SavedPipeline[]>([])
  const [pipeline, setPipeline] = useState<Pipeline>(emptyPipeline)
  const [runs, setRuns] = useState<Run[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)

  const [booting, setBooting] = useState(true)
  const [bootError, setBootError] = useState<string | null>(null)
  const [libLoading, setLibLoading] = useState(true)
  const [libError, setLibError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [running, setRunning] = useState(false)
  const [connected, setConnected] = useState(true)
  const [runError, setRunError] = useState<string | null>(null)

  const unsub = useRef<(() => void) | null>(null)

  const refreshLibrary = useCallback(async () => {
    setLibLoading(true)
    try {
      setSaved(await api.listPipelines())
      setLibError(null)
    } catch (e) {
      setLibError(describe(e))
    } finally {
      setLibLoading(false)
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const [a, r] = await Promise.all([api.listAgents(), api.listRuns()])
        setAgents(a)
        setRuns(r)
        if (r.length) setActiveId(r[0].run_id)
        setBootError(null)
      } catch (e) {
        setBootError(describe(e))
      } finally {
        setBooting(false)
      }
    })()
    refreshLibrary()
    return () => unsub.current?.()
  }, [refreshLibrary])

  const watch = useCallback((id: string) => {
    unsub.current?.()
    setConnected(true)
    unsub.current = api.subscribeRun(
      id,
      (e) => setRuns((prev) => prev.map((r) => (r.run_id === id ? applyEvent(r, e) : r))),
      (reason) => {
        setConnected(reason === 'done')
        if (reason === 'done') setRunning(false)
      },
    )
  }, [])

  const handleRun = async (req: RunRequest) => {
    setRunning(true)
    setRunError(null)
    try {
      const started = await api.startRun(toYaml(pipeline, agents), req)
      if (!started.length) {
        setRunError('No resource matched that input. Check the path or glob, then run again.')
        setRunning(false)
        return
      }
      setRuns((prev) => [...started, ...prev])
      setActiveId(started[0].run_id)
      setView('runs')
      // Only the selected run streams; the others fill in from their own Run objects.
      watch(started[0].run_id)
    } catch (e) {
      setRunError(describe(e))
      setRunning(false)
    }
  }

  const handleCancel = async (id: string) => {
    unsub.current?.()
    unsub.current = null
    try {
      const run = await api.cancelRun(id)
      // Keep the locally folded steps/events: they are at least as new as this snapshot.
      setRuns((prev) =>
        prev.map((r) => (r.run_id === id ? { ...r, status: run.status, ended_at: run.ended_at, error: run.error } : r)),
      )
    } catch (e) {
      setRunError(describe(e))
    }
  }

  const handleSave = async (name: string) => {
    setSaving(true)
    try {
      await api.savePipeline(name, toYaml(pipeline, agents))
      await refreshLibrary()
      setLibError(null)
    } catch (e) {
      setLibError(describe(e))
    } finally {
      setSaving(false)
    }
  }

  const handleOpen = async (name: string) => {
    setBusy(name)
    try {
      const { yaml } = await api.readPipeline(name)
      setPipeline(fromYaml(yaml))
      setView('builder')
      setLibError(null)
    } catch (e) {
      setLibError(describe(e))
    } finally {
      setBusy(null)
    }
  }

  const handleDelete = async (name: string) => {
    setBusy(name)
    try {
      await api.deletePipeline(name)
      await refreshLibrary()
    } catch (e) {
      setLibError(describe(e))
    } finally {
      setBusy(null)
    }
  }

  const handleImport = (name: string, text: string) => {
    try {
      const next = fromYaml(text)
      setPipeline({ ...next, pipeline_id: next.pipeline_id || name })
      setView('builder')
      setLibError(null)
    } catch (e) {
      setLibError(`${name}.yaml could not be read. ${describe(e)}`)
    }
  }

  const activeRun = runs.find((r) => r.run_id === activeId) ?? null
  const activeStatus = activeRun?.status

  // Depend on primitives: applyEvent returns a new run object per event, and depending on
  // that object would tear down and reopen the EventSource on every single event.
  useEffect(() => {
    if (activeId && (activeStatus === 'running' || activeStatus === 'queued')) watch(activeId)
  }, [activeId, activeStatus, watch])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <header
        style={{
          height: 56,
          flexShrink: 0,
          borderBottom: '1px solid var(--c22)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 20px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
          <div style={{ fontFamily: 'var(--serif)', fontSize: 19, fontWeight: 500, letterSpacing: '0.04em' }}>OmniIngest</div>
          <div style={{ width: 1, height: 22, background: 'var(--c22)' }} />
          <nav style={{ display: 'flex', gap: 20 }}>
            {(['builder', 'runs', 'library'] as View[]).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                style={{
                  border: 0,
                  padding: '0 0 2px',
                  fontFamily: 'var(--mono)',
                  fontSize: 10,
                  letterSpacing: '0.16em',
                  textTransform: 'uppercase',
                  color: view === v ? 'var(--cream)' : 'var(--c50)',
                  borderBottom: view === v ? '2px solid var(--accent)' : '2px solid transparent',
                }}
              >
                {v}
              </button>
            ))}
          </nav>
        </div>
        <div className="label" style={{ letterSpacing: '0.16em' }}>
          {agents.length} AGENTS &middot; {runs.filter((r) => r.status === 'running').length} RUNNING
        </div>
      </header>

      {bootError && (
        <div
          style={{
            borderBottom: '1px solid var(--failed)',
            padding: '12px 24px',
            fontFamily: 'var(--mono)',
            fontSize: 10,
            lineHeight: 1.7,
            color: 'var(--failed)',
          }}
        >
          {bootError}
        </div>
      )}

      <main className="view-host" style={{ flexGrow: 1, minHeight: 0, display: 'flex' }}>
        {booting ? (
          <div className="empty" style={{ margin: 'auto' }}>
            Loading the agent registry from the server&hellip;
          </div>
        ) : view === 'builder' ? (
          <Builder
            agents={agents}
            pipeline={pipeline}
            onChange={setPipeline}
            onRun={handleRun}
            onSave={handleSave}
            running={running}
            saving={saving}
            runError={runError}
          />
        ) : view === 'runs' ? (
          <Runs runs={runs} activeId={activeId} onSelect={setActiveId} onCancel={handleCancel} connected={connected} error={runError} />
        ) : (
          <Library
            items={saved}
            loading={libLoading}
            error={libError}
            busy={busy}
            onOpen={handleOpen}
            onDelete={handleDelete}
            onImport={handleImport}
            onNew={() => {
              setPipeline({ ...emptyPipeline(), steps: [] })
              setView('builder')
            }}
          />
        )}
      </main>
    </div>
  )
}
