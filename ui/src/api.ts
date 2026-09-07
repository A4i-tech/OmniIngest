import type { AgentSpec, Run, RunEvent, RunRequest, SavedPipeline } from './types'

const BASE = '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly hint: string,
  ) {
    super(message)
  }
}

/** Python serializes an absent optional as null; the TS types say undefined. Reconcile once, here. */
const dropNulls = (_key: string, value: unknown) => (value === null ? undefined : value)

export const parseJson = <T,>(text: string): T => JSON.parse(text, dropNulls) as T

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(BASE + path, {
      ...init,
      headers: init?.body ? { 'content-type': 'application/json' } : undefined,
    })
  } catch {
    throw new ApiError(
      'Cannot reach the OmniIngest server.',
      'Start it with `uv run --project server uvicorn app:app --port 8077` from omniingest-ui, then reload.',
    )
  }
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new ApiError(
      `Server refused ${init?.method ?? 'GET'} ${path} (HTTP ${res.status}).`,
      detail.slice(0, 400) || 'Check the server log for the traceback.',
    )
  }
  const body = await res.text()
  try {
    return parseJson<T>(body)
  } catch {
    throw new ApiError(
      `${path} returned a success code but not JSON.`,
      'A proxy or dev server is probably answering instead of the OmniIngest server. Check that it is running on port 8077 and reload.',
    )
  }
}

export const listAgents = () => req<AgentSpec[]>('/agents')
export const listPipelines = () => req<SavedPipeline[]>('/pipelines')
export const readPipeline = (name: string) => req<{ yaml: string }>(`/pipelines/${encodeURIComponent(name)}`)

export const savePipeline = (name: string, text: string) =>
  req<SavedPipeline>(`/pipelines/${encodeURIComponent(name)}`, { method: 'PUT', body: JSON.stringify({ yaml: text }) })

export const deletePipeline = (name: string) =>
  req<{ ok: true }>(`/pipelines/${encodeURIComponent(name)}`, { method: 'DELETE' })

export const listRuns = () => req<Run[]>('/runs')
export const readRun = (id: string) => req<Run>(`/runs/${id}`)

/** One resolved resource becomes one run, so a glob or directory returns several. */
export const startRun = (yaml: string, run: RunRequest) =>
  req<Run[]>('/runs', {
    method: 'POST',
    body: JSON.stringify({
      yaml,
      input: run.inputs,
      input_name: run.inputName,
      output: run.output,
      format: run.format,
      params: run.params,
    }),
  })

export const resumeRun = (id: string, yaml: string) =>
  req<Run>(`/runs/${id}/resume`, { method: 'POST', body: JSON.stringify({ yaml }) })

export const previewResources = (path: string) =>
  req<{ resources: string[]; count: number }>(`/resources?path=${encodeURIComponent(path)}`)

export const cancelRun = (id: string) => req<Run>(`/runs/${id}/cancel`, { method: 'POST' })

/** Live events for one run. Returns an unsubscribe function. */
export function subscribeRun(id: string, onEvent: (e: RunEvent) => void, onClose?: (reason: 'done' | 'error') => void) {
  const es = new EventSource(`${BASE}/runs/${id}/events`)
  es.onmessage = (m) => {
    let event: RunEvent
    try {
      event = parseJson<RunEvent>(m.data)
    } catch {
      // A dropped event silently stales the run, so surface it as a lost connection.
      es.close()
      onClose?.('error')
      return
    }
    onEvent(event)
    if (event.kind === 'pipeline_end') {
      es.close()
      onClose?.('done')
    }
  }
  es.onerror = () => {
    es.close()
    onClose?.('error')
  }
  return () => es.close()
}
