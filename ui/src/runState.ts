import type { Run, RunEvent, RunStatus } from './types'

/** Fold one live event into a run so the UI does not refetch on every tick. */
export function applyEvent(run: Run, e: RunEvent): Run {
  const events = [...run.events, e]
  const steps = run.steps.map((s) => ({ ...s }))
  const target = e.step_index !== undefined ? steps[e.step_index] : undefined

  if (target) {
    if (e.kind === 'step_begin') {
      target.status = 'running'
      target.start = e.t
    } else if (e.kind === 'step_progress') {
      target.status = 'running'
      target.completed = e.completed
      target.total = e.total
      if (target.start === undefined) target.start = e.t
    } else if (e.kind === 'step_end') {
      target.status = e.status === 'failure' || e.status === 'skipped' ? e.status : 'success'
      target.end = e.t
      if (target.start === undefined) target.start = e.t
    }
  }

  let status: RunStatus = run.status
  let ended_at = run.ended_at
  let error = run.error
  if (e.kind === 'pipeline_end') {
    // The backend sends succeeded/failed here, not the step vocabulary.
    status = e.status === 'succeeded' ? 'succeeded' : 'failed'
    ended_at = run.started_at + e.t
    error = e.error ?? run.error
    // A step still marked running when the pipeline ends never reported its end.
    for (const s of steps) if (s.status === 'running') s.status = status === 'failed' ? 'failure' : 'success'
  }

  return { ...run, events, steps, status, ended_at, error }
}
