import { useState } from 'react'
import type { CSSProperties } from 'react'
import type { Pipeline } from '../types'

interface PipelineSettingsProps {
  pipeline: Pipeline
  onChange: (next: Pipeline) => void
  onClose: () => void
}

type ParamType = 'string' | 'integer' | 'number' | 'boolean' | 'array' | 'object'
const PARAM_TYPES: ParamType[] = ['string', 'integer', 'number', 'boolean', 'array', 'object']
const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/

interface ParamSchema {
  type: ParamType
  default?: unknown
  description?: string
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function isParamType(v: unknown): v is ParamType {
  return typeof v === 'string' && (PARAM_TYPES as string[]).includes(v)
}

function toParamSchema(v: unknown): ParamSchema {
  if (!isRecord(v)) return { type: 'string' }
  return {
    type: isParamType(v.type) ? v.type : 'string',
    default: v.default,
    description: typeof v.description === 'string' ? v.description : undefined,
  }
}

function getEntries(pipeline: Pipeline): [string, ParamSchema][] {
  const props = pipeline.parameters.properties
  if (!isRecord(props)) return []
  return Object.entries(props).map(([name, schema]) => [name, toParamSchema(schema)])
}

function getRequired(pipeline: Pipeline): string[] {
  const req = pipeline.parameters.required
  return Array.isArray(req) ? req.filter((x): x is string => typeof x === 'string') : []
}

function uniqueName(existing: string[]): string {
  let i = 1
  while (existing.includes(`param_${i}`)) i++
  return `param_${i}`
}

// integer/number/boolean/array/object each carry a JS-typed default; an empty default means "unset", always valid
function defaultInvalid(type: ParamType, value: unknown): boolean {
  if (value === undefined || value === '') return false
  switch (type) {
    case 'integer':
      return typeof value !== 'number' || !Number.isInteger(value)
    case 'number':
      return typeof value !== 'number' || Number.isNaN(value)
    case 'boolean':
      return typeof value !== 'boolean'
    case 'array':
      return !Array.isArray(value)
    case 'object':
      return !isRecord(value)
    default:
      return false
  }
}

function writeParameters(pipeline: Pipeline, onChange: (p: Pipeline) => void, entries: [string, ParamSchema][], required: string[]) {
  const properties: Record<string, unknown> = {}
  for (const [name, schema] of entries) {
    const frag: Record<string, unknown> = { type: schema.type }
    if (schema.default !== undefined && schema.default !== '') frag.default = schema.default
    if (schema.description) frag.description = schema.description
    properties[name] = frag
  }
  onChange({ ...pipeline, parameters: { ...pipeline.parameters, properties, required } })
}

function updateParam(
  pipeline: Pipeline,
  onChange: (p: Pipeline) => void,
  index: number,
  patch: { name?: string; schema?: ParamSchema },
) {
  const entries = getEntries(pipeline)
  const required = getRequired(pipeline)
  const [oldName, oldSchema] = entries[index]
  const newName = patch.name ?? oldName
  entries[index] = [newName, patch.schema ?? oldSchema]
  const nextRequired = patch.name !== undefined ? required.map((r) => (r === oldName ? newName : r)) : required
  writeParameters(pipeline, onChange, entries, nextRequired)
}

function deleteParam(pipeline: Pipeline, onChange: (p: Pipeline) => void, index: number) {
  const entries = getEntries(pipeline)
  const [name] = entries[index]
  entries.splice(index, 1)
  writeParameters(pipeline, onChange, entries, getRequired(pipeline).filter((r) => r !== name))
}

function addParam(pipeline: Pipeline, onChange: (p: Pipeline) => void) {
  const entries = getEntries(pipeline)
  entries.push([uniqueName(entries.map(([n]) => n)), { type: 'string', default: '' }])
  writeParameters(pipeline, onChange, entries, getRequired(pipeline))
}

function toggleRequired(pipeline: Pipeline, onChange: (p: Pipeline) => void, name: string, isRequired: boolean) {
  const required = getRequired(pipeline)
  const next = isRequired ? [...required, name] : required.filter((r) => r !== name)
  writeParameters(pipeline, onChange, getEntries(pipeline), next)
}

function XIcon({ size = 11 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 11 11" fill="none">
      <path d="M1 1L10 10M10 1L1 10" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  )
}

const helpStyle: CSSProperties = { fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 4 }
const errStyle: CSSProperties = { fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--failed)', marginTop: 4 }

export default function PipelineSettings({ pipeline, onChange, onClose }: PipelineSettingsProps) {
  const [defaultDraft, setDefaultDraft] = useState<Record<number, string>>({})

  const entries = getEntries(pipeline)
  const required = getRequired(pipeline)
  const seen = new Set<string>()

  return (
    <div style={{ width: 480, height: '100%', background: 'var(--ground)', borderLeft: '1px solid var(--c22)', display: 'flex', flexDirection: 'column' }}>
      <style>{`
        .ps-x { border: none; background: none; padding: 4px; color: var(--c50); display: flex; cursor: pointer; }
        .ps-x:hover { color: var(--cream); }
        .ps-required { width: 14px; height: 14px; padding: 0; border: none; flex-shrink: 0; accent-color: var(--accent); }
      `}</style>
      <div style={{ height: 56, borderBottom: '1px solid var(--c22)', padding: '0 20px', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ fontFamily: 'var(--serif)', fontSize: 22 }}>Pipeline settings</div>
        <button className="ps-x" onClick={onClose}>
          <XIcon />
        </button>
      </div>

      <div className="scroll" style={{ flex: 1 }}>
        <div className="label" style={{ padding: '18px 20px 10px' }}>IDENTITY</div>
        <div style={{ padding: '0 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <div className="label-sm" style={{ marginBottom: 4 }}>PIPELINE ID</div>
            <input placeholder="kseeb_lba_questions" value={pipeline.pipeline_id} onChange={(e) => onChange({ ...pipeline, pipeline_id: e.target.value })} />
          </div>
          <div>
            <div className="label-sm" style={{ marginBottom: 4 }}>DOMAIN PROFILE</div>
            <input placeholder="default" value={pipeline.domain_profile} onChange={(e) => onChange({ ...pipeline, domain_profile: e.target.value })} />
            <div style={helpStyle}>Used when one store holds several domains.</div>
          </div>
          <div>
            <div className="label-sm" style={{ marginBottom: 4 }}>INPUT MODALITY</div>
            <select value={pipeline.input_modality} onChange={(e) => onChange({ ...pipeline, input_modality: e.target.value })}>
              <option value=""></option>
              <option value="text">text</option>
              <option value="document">document</option>
              <option value="audio">audio</option>
            </select>
            <div style={helpStyle}>A human hint. The runner does not use it.</div>
          </div>
          <div>
            <div className="label-sm" style={{ marginBottom: 4 }}>DESCRIPTION</div>
            <textarea rows={2} value={pipeline.description} onChange={(e) => onChange({ ...pipeline, description: e.target.value })} />
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '18px 20px 10px' }}>
          <div className="label">PARAMETERS</div>
          <button className="btn btn-sm" onClick={() => addParam(pipeline, onChange)}>+ PARAMETER</button>
        </div>
        <div style={{ padding: '0 20px 4px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {entries.length === 0 && <div className="empty">No parameters. Add one to make a value configurable per run.</div>}
          {entries.map(([name, schema], index) => {
            const nameValid = NAME_RE.test(name)
            const isDup = nameValid && seen.has(name)
            seen.add(name)
            const nameError = !nameValid
              ? 'Use letters, digits and underscores, starting with a letter or underscore. This name becomes ${.config.<name>}.'
              : isDup
                ? 'A parameter with this name already exists. Rename one of them.'
                : undefined
            const isRequired = required.includes(name)
            const invalidDefault = defaultInvalid(schema.type, schema.default)
            const draft = defaultDraft[index]

            return (
              <div key={index} style={{ borderBottom: '1px solid var(--c18)', paddingBottom: 12 }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                  <input
                    style={{ flex: 1, ...(nameError ? { borderColor: 'var(--failed)' } : {}) }}
                    value={name}
                    onChange={(e) => updateParam(pipeline, onChange, index, { name: e.target.value })}
                  />
                  <select
                    style={{ width: 92 }}
                    value={schema.type}
                    onChange={(e) => {
                      const type = isParamType(e.target.value) ? e.target.value : 'string'
                      updateParam(pipeline, onChange, index, { schema: { ...schema, type } })
                    }}
                  >
                    {PARAM_TYPES.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                  <button className="ps-x" onClick={() => deleteParam(pipeline, onChange, index)}>
                    <XIcon size={9} />
                  </button>
                </div>
                {nameError && <div style={errStyle}>{nameError}</div>}

                <div style={{ marginTop: 8 }}>
                  {schema.type === 'boolean' ? (
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button
                        className={schema.default === true ? 'btn btn-primary' : 'btn'}
                        style={{ flex: 1 }}
                        onClick={() => updateParam(pipeline, onChange, index, { schema: { ...schema, default: true } })}
                      >
                        TRUE
                      </button>
                      <button
                        className={schema.default === false ? 'btn btn-primary' : 'btn'}
                        style={{ flex: 1 }}
                        onClick={() => updateParam(pipeline, onChange, index, { schema: { ...schema, default: false } })}
                      >
                        FALSE
                      </button>
                    </div>
                  ) : schema.type === 'integer' || schema.type === 'number' ? (
                    <input
                      type="number"
                      step={schema.type === 'integer' ? 1 : undefined}
                      placeholder="default"
                      style={invalidDefault ? { borderColor: 'var(--failed)' } : {}}
                      value={typeof schema.default === 'number' ? schema.default : ''}
                      onChange={(e) =>
                        updateParam(pipeline, onChange, index, {
                          schema: { ...schema, default: e.target.value === '' ? '' : Number(e.target.value) },
                        })
                      }
                    />
                  ) : schema.type === 'array' || schema.type === 'object' ? (
                    <input
                      placeholder={schema.type === 'array' ? '[]' : '{}'}
                      style={invalidDefault ? { borderColor: 'var(--failed)' } : {}}
                      value={draft ?? (schema.default === undefined || schema.default === '' ? '' : JSON.stringify(schema.default))}
                      onChange={(e) => {
                        const text = e.target.value
                        setDefaultDraft((d) => ({ ...d, [index]: text }))
                        if (text.trim() === '') {
                          updateParam(pipeline, onChange, index, { schema: { ...schema, default: '' } })
                          return
                        }
                        try {
                          const parsed: unknown = JSON.parse(text)
                          const ok = schema.type === 'array' ? Array.isArray(parsed) : isRecord(parsed)
                          if (ok) updateParam(pipeline, onChange, index, { schema: { ...schema, default: parsed } })
                        } catch {
                          // invalid JSON mid-typing: keep the draft text, invalidDefault below flags it once committed value is stale
                        }
                      }}
                      onBlur={() =>
                        setDefaultDraft((d) => {
                          const next = { ...d }
                          delete next[index]
                          return next
                        })
                      }
                    />
                  ) : (
                    <input
                      placeholder="default"
                      value={typeof schema.default === 'string' ? schema.default : ''}
                      onChange={(e) => updateParam(pipeline, onChange, index, { schema: { ...schema, default: e.target.value } })}
                    />
                  )}
                  {invalidDefault && <div style={errStyle}>This default is not a valid {schema.type}. The run rejects the pipeline before any step executes.</div>}
                </div>

                <input
                  style={{ marginTop: 8 }}
                  placeholder="description"
                  value={schema.description ?? ''}
                  onChange={(e) => updateParam(pipeline, onChange, index, { schema: { ...schema, description: e.target.value } })}
                />

                <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)' }}>
                  <input
                    type="checkbox"
                    className="ps-required"
                    checked={isRequired}
                    onChange={(e) => toggleRequired(pipeline, onChange, name, e.target.checked)}
                  />
                  REQUIRED
                </label>

                <div className="mono" style={{ fontSize: 9, color: 'var(--jq)', marginTop: 8, userSelect: 'text' }}>
                  {'${.config.' + name + '}'}
                </div>
              </div>
            )
          })}
        </div>

        <div className="label" style={{ padding: '18px 20px 10px' }}>TOOL VENDORS</div>
        <div style={{ padding: '0 20px 20px', ...helpStyle, marginTop: 0 }}>
          MCP tool vendors are defined in the YAML and are not editable here yet. Each agent&apos;s toolsets field references a vendor by name.
        </div>
      </div>
    </div>
  )
}
