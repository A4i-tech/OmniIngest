import yaml from 'js-yaml'
import type { AgentSpec, FieldSpec, Pipeline, StepNode, ValidationCheck } from './types'

let seq = 0
export const newUid = () => `n${++seq}${Date.now().toString(36).slice(-4)}`

export function emptyPipeline(): Pipeline {
  return {
    pipeline_id: '',
    domain_profile: '',
    input_modality: '',
    description: '',
    parameters: {},
    steps: [],
  }
}

const blankFor = (type: FieldSpec['type']): unknown => {
  if (type === 'object') return {}
  if (type === 'array') return []
  if (type === 'boolean') return false
  if (type === 'integer' || type === 'number') return 0
  return ''
}

export function newStep(spec: AgentSpec): StepNode {
  const config: Record<string, unknown> = {}
  // Required fields have no default to fall back on, so seed one of the right shape.
  for (const f of spec.fields) if (f.required) config[f.name] = f.default ?? blankFor(f.type)
  return { uid: newUid(), agent: spec.id, config, validation: [] }
}

const isBlank = (v: unknown) =>
  v === undefined ||
  v === null ||
  v === '' ||
  (Array.isArray(v) && v.length === 0) ||
  (typeof v === 'object' && !Array.isArray(v) && Object.keys(v as object).length === 0)

const sameValue = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/** Drop keys that match the agent's declared default — the backend applies them anyway. */
export function pruneConfig(config: Record<string, unknown>, spec?: AgentSpec): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(config)) {
    const field = spec?.fields.find((f) => f.name === k)
    if (isBlank(v) && !field?.required) continue
    if (field && !field.required && field.default !== undefined && sameValue(v, field.default)) continue
    out[k] = v
  }
  return out
}

export function toYaml(pipeline: Pipeline, specs: AgentSpec[] = []): string {
  const byId = new Map(specs.map((s) => [s.id, s]))
  const doc: Record<string, unknown> = {}

  if (pipeline.pipeline_id) doc.pipeline_id = pipeline.pipeline_id
  if (pipeline.domain_profile) doc.domain_profile = pipeline.domain_profile
  if (pipeline.input_modality) doc.input_modality = pipeline.input_modality
  if (pipeline.description) doc.description = pipeline.description
  if (!isBlank(pipeline.parameters)) doc.parameters = pipeline.parameters

  doc.steps = pipeline.steps.map((s) => {
    const step: Record<string, unknown> = { agent: s.agent }
    if (s.description) step.description = s.description
    if (s.stage) step.stage = s.stage
    const config = pruneConfig(s.config, byId.get(s.agent))
    if (!isBlank(config)) step.config = config
    if (s.validation.length) step.validation = s.validation
    return step
  })

  return yaml.dump(doc, { lineWidth: 96, noRefs: true, quotingType: '"', indent: 2 })
}

export function fromYaml(text: string): Pipeline {
  const doc = yaml.load(text)
  if (!doc || typeof doc !== 'object' || Array.isArray(doc)) throw new Error('YAML root must be a mapping with a `steps` list.')
  const raw = doc as Record<string, unknown>
  if (!Array.isArray(raw.steps)) throw new Error('YAML has no `steps` list. Every pipeline needs one.')

  const steps: StepNode[] = raw.steps.map((s, i) => {
    if (!s || typeof s !== 'object') throw new Error(`Step ${i + 1} is not a mapping.`)
    const step = s as Record<string, unknown>
    if (typeof step.agent !== 'string') throw new Error(`Step ${i + 1} has no \`agent\` name.`)
    return {
      uid: newUid(),
      agent: step.agent,
      description: typeof step.description === 'string' ? step.description : undefined,
      stage: typeof step.stage === 'string' ? step.stage : undefined,
      config: (step.config as Record<string, unknown>) ?? {},
      validation: (step.validation as ValidationCheck[]) ?? [],
    }
  })

  return {
    pipeline_id: typeof raw.pipeline_id === 'string' ? raw.pipeline_id : '',
    domain_profile: typeof raw.domain_profile === 'string' ? raw.domain_profile : '',
    input_modality: typeof raw.input_modality === 'string' ? raw.input_modality : '',
    description: typeof raw.description === 'string' ? raw.description : '',
    parameters: (raw.parameters as Record<string, unknown>) ?? {},
    steps,
  }
}

/** Contiguous runs of steps sharing a `stage` render as one dashed group box. */
export interface StepGroup {
  stage?: string
  steps: StepNode[]
}

export function groupByStage(steps: StepNode[]): StepGroup[] {
  const groups: StepGroup[] = []
  for (const step of steps) {
    const last = groups[groups.length - 1]
    if (step.stage && last && last.stage === step.stage) last.steps.push(step)
    else groups.push({ stage: step.stage, steps: [step] })
  }
  return groups
}

/** The backend rejects a stage split across non-adjacent steps, so catch it before running. */
export function validatePipeline(pipeline: Pipeline, specs: AgentSpec[] = []): string[] {
  const errors: string[] = []
  const known = new Set(specs.map((s) => s.id))

  if (!pipeline.steps.length) errors.push('Pipeline has no steps. Drag an agent from the library to start.')

  const seen = new Set<string>()
  let prev: string | undefined
  for (const step of pipeline.steps) {
    if (step.stage) {
      if (step.stage !== prev && seen.has(step.stage)) {
        errors.push(`Stage "${step.stage}" is split into two blocks. Move its steps next to each other — stages must be contiguous.`)
      }
      seen.add(step.stage)
    }
    prev = step.stage
  }

  for (const group of groupByStage(pipeline.steps)) {
    if (!group.stage) continue
    const clashing = group.steps.filter((s) => specs.find((a) => a.id === s.agent)?.replaces_items)
    if (clashing.length > 1) {
      errors.push(
        `Stage "${group.stage}" runs ${clashing.map((s) => s.agent).join(' and ')} together, and each one rebuilds the item list. ` +
          `They would overwrite each other. Split them into separate stages, or keep only one.`,
      )
    }
  }

  for (const [i, step] of pipeline.steps.entries()) {
    if (specs.length && !known.has(step.agent)) {
      errors.push(`Step ${i + 1} uses unknown agent "${step.agent}". Pick one from the library.`)
      continue
    }
    const spec = specs.find((s) => s.id === step.agent)
    for (const f of spec?.fields ?? []) {
      if (f.required && isBlank(step.config[f.name])) {
        errors.push(`Step ${i + 1} (${step.agent}) needs a value for "${f.name}". ${f.description ?? ''}`.trim())
      }
    }
  }
  return errors
}
