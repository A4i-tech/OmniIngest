export type FieldType = 'string' | 'text' | 'jq' | 'integer' | 'number' | 'boolean' | 'enum' | 'array' | 'object'

export interface FieldSpec {
  name: string
  type: FieldType
  required: boolean
  default?: unknown
  description?: string
  choices?: string[]
}

/** One entry in the palette. `primitive` = a registered Step class, `component` = a profiles/*.yaml. */
export interface AgentSpec {
  id: string
  cls: string
  kind: 'primitive' | 'component'
  group: string
  description?: string
  fields: FieldSpec[]
  /** component only: the agent ids of its inner steps, and where the yaml lives */
  steps?: string[]
  path?: string
  /** rebuilds ctx.items wholesale, so two of these must never share a stage */
  replaces_items?: boolean
}

export type ValidationMode = 'condition' | 'before' | 'plain' | 'after'

export interface ValidationCheck {
  mode: ValidationMode
  expr: string
  message: string
}

/** A step in the editor. `uid` is client-side identity so React keys survive reordering. */
export interface StepNode {
  uid: string
  agent: string
  description?: string
  stage?: string
  config: Record<string, unknown>
  validation: ValidationCheck[]
}

export interface Pipeline {
  pipeline_id: string
  domain_profile: string
  input_modality: string
  description: string
  parameters: Record<string, unknown>
  steps: StepNode[]
}

export interface SavedPipeline {
  name: string
  path: string
  source: 'profiles' | 'tests' | 'local'
  pipeline_id?: string
  description?: string
  agents: string[]
  modified?: number
}

export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type StepStatus = 'pending' | 'running' | 'success' | 'failure' | 'skipped'

export interface RunEvent {
  /** milliseconds since the run began */
  t: number
  kind: 'pipeline_begin' | 'step_begin' | 'step_progress' | 'step_end' | 'pipeline_end'
  step?: string
  step_index?: number
  stage?: string
  completed?: number
  total?: number
  message?: string
  /** step_end carries a StepStatus; pipeline_end carries succeeded/failed. */
  status?: StepStatus | 'succeeded' | 'failed'
  error?: string
}

export interface RunStepState {
  agent: string
  stage?: string
  status: StepStatus
  /** ms offsets from run start; end is undefined while running */
  start?: number
  end?: number
  completed?: number
  total?: number
}

export interface Run {
  run_id: string
  pipeline_id: string
  source: string
  output: string
  tenant: string
  status: RunStatus
  started_at: number
  ended_at?: number
  error?: string
  steps: RunStepState[]
  events: RunEvent[]
}

/** What the run dialog assembles and the backend expects. Mirrors the CLI flags. */
export interface RunRequest {
  inputs: string[]
  inputName?: string
  output: string
  format?: 'json' | 'pkl'
  params: Record<string, unknown>
}
