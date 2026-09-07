import { useEffect, useState } from 'react'
import type { Pipeline } from '../types'

export interface RunDialogProps {
  pipeline: Pipeline
  busy: boolean
  error: string | null
  onSubmit: (req: {
    inputs: string[]
    inputName?: string
    output: string
    format?: 'json' | 'pkl'
    params: Record<string, unknown>
  }) => void
  onClose: () => void
}

type OutputKind = 'discard' | 'json' | 'pkl' | 'vector' | 'graph' | 'mongodb'

interface PropertySchema {
  type: string
  description?: string
  default?: unknown
  enum?: string[]
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

// mirrors cli.py: a schema type can be a list like ["string", "null"]
function resolveType(raw: unknown): string {
  if (typeof raw === 'string') return raw
  if (Array.isArray(raw)) {
    const found = raw.find((t) => t !== 'null')
    if (typeof found === 'string') return found
  }
  return 'string'
}

function asPropertySchema(v: unknown): PropertySchema {
  if (!isRecord(v)) return { type: 'string' }
  return {
    type: resolveType(v.type),
    description: typeof v.description === 'string' ? v.description : undefined,
    default: v.default,
    enum: Array.isArray(v.enum) ? v.enum.filter((e): e is string => typeof e === 'string') : undefined,
  }
}

function getProperties(parameters: Record<string, unknown>): Record<string, PropertySchema> {
  const properties = isRecord(parameters.properties) ? parameters.properties : {}
  const out: Record<string, PropertySchema> = {}
  for (const [name, schema] of Object.entries(properties)) out[name] = asPropertySchema(schema)
  return out
}

function getRequired(parameters: Record<string, unknown>): string[] {
  return Array.isArray(parameters.required) ? parameters.required.filter((r): r is string => typeof r === 'string') : []
}

function XIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9">
      <line x1="1" y1="1" x2="8" y2="8" stroke="currentColor" strokeWidth="1.2" />
      <line x1="8" y1="1" x2="1" y2="8" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  )
}

let nextRowId = 0

export default function RunDialog({ pipeline, busy, error, onSubmit, onClose }: RunDialogProps) {
  const properties = getProperties(pipeline.parameters)
  const required = getRequired(pipeline.parameters)
  const propertyNames = Object.keys(properties)

  const [inputs, setInputs] = useState<{ id: number; value: string }[]>([{ id: nextRowId++, value: '' }])
  const [inputName, setInputName] = useState('')
  const [outputKind, setOutputKind] = useState<OutputKind>('discard')
  const [filePath, setFilePath] = useState('out.json')
  const [mongoUri, setMongoUri] = useState('')
  const [format, setFormat] = useState<'auto' | 'json' | 'pkl'>('auto')
  const [params, setParams] = useState<Record<string, unknown>>(() => {
    const init: Record<string, unknown> = {}
    for (const name of propertyNames) {
      const schema = properties[name]
      if ('default' in schema) init[name] = schema.default
    }
    return init
  })

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  function addInput() {
    setInputs((rows) => [...rows, { id: nextRowId++, value: '' }])
  }
  function removeInput(id: number) {
    setInputs((rows) => (rows.length > 1 ? rows.filter((row) => row.id !== id) : rows))
  }
  function updateInput(id: number, value: string) {
    setInputs((rows) => rows.map((row) => (row.id === id ? { ...row, value } : row)))
  }

  const filledInputs = inputs.filter((row) => row.value.trim() !== '')
  const hasStdin = filledInputs.some((row) => row.value.trim() === '-')
  const stdinError =
    hasStdin && filledInputs.length > 1 ? 'Stdin cannot be combined with another input. Remove the others, or remove the dash.' : null

  let outputValue = ''
  if (outputKind === 'discard') outputValue = '-'
  else if (outputKind === 'json' || outputKind === 'pkl') outputValue = filePath
  else if (outputKind === 'vector') outputValue = 'vector:default'
  else if (outputKind === 'graph') outputValue = 'graph:default'
  else outputValue = mongoUri

  const outputError =
    outputKind === 'mongodb' && !(mongoUri.startsWith('mongodb://') || mongoUri.startsWith('mongodb+srv://'))
      ? 'A MongoDB target must start with mongodb:// or mongodb+srv://.'
      : (outputKind === 'json' || outputKind === 'pkl') && filePath.trim() === ''
        ? 'Give the output file a path, for example out.json.'
        : null

  const paramError = required.some((name) => {
    const value = params[name]
    return value === undefined || value === null || value === ''
  })
    ? 'This parameter is required. The run is rejected before any step executes.'
    : null

  const validationMessage = stdinError ?? outputError ?? paramError
  const submitDisabled = busy || validationMessage !== null || filledInputs.length === 0

  function handleSubmit() {
    if (submitDisabled) return
    onSubmit({
      inputs: filledInputs.map((row) => row.value.trim()),
      inputName: hasStdin && inputName.trim() !== '' ? inputName.trim() : undefined,
      output: outputValue,
      format: format === 'auto' ? undefined : format,
      params,
    })
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(8, 11, 22, 0.72)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 620,
          maxHeight: '82vh',
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--ground)',
          border: '1px solid var(--c22)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            height: 52,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 20px',
            borderBottom: '1px solid var(--c18)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <span style={{ fontFamily: 'var(--serif)', fontSize: 22 }}>Run pipeline</span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--c62)' }}>{pipeline.pipeline_id}</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="run-dialog-x"
            style={{ border: 'none', background: 'transparent', padding: 0, color: 'var(--c50)', display: 'flex' }}
          >
            <XIcon />
          </button>
        </div>

        <div className="scroll" style={{ flex: 1, padding: 20, display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div>
            <div className="label" style={{ marginBottom: 8 }}>
              Inputs
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {inputs.map((row) => (
                <div key={row.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    value={row.value}
                    onChange={(e) => updateInput(row.id, e.target.value)}
                    placeholder="path, glob, directory, http(s) URL, or - for stdin"
                  />
                  <button
                    type="button"
                    onClick={() => removeInput(row.id)}
                    aria-label="Remove input"
                    className="run-dialog-x"
                    style={{ border: 'none', background: 'transparent', padding: 0, color: 'var(--c50)', display: 'flex', flexShrink: 0 }}
                  >
                    <XIcon />
                  </button>
                </div>
              ))}
            </div>
            <button type="button" className="btn btn-sm" style={{ marginTop: 8 }} onClick={addInput}>
              + INPUT
            </button>
            <div style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 8 }}>
              A glob or a directory becomes one run per matched resource.
            </div>
            {hasStdin && (
              <div style={{ marginTop: 12 }}>
                <div className="label-sm" style={{ marginBottom: 6 }}>
                  Input name
                </div>
                <input value={inputName} onChange={(e) => setInputName(e.target.value)} placeholder="filename hint" />
                <div style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 6 }}>
                  A filename hint for stdin, so the parser can pick a format.
                </div>
              </div>
            )}
            {stdinError && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)', marginTop: 8 }}>{stdinError}</div>
            )}
          </div>

          <div>
            <div className="label" style={{ marginBottom: 8 }}>
              Output
            </div>
            <select value={outputKind} onChange={(e) => setOutputKind(e.target.value as OutputKind)}>
              <option value="discard">Discard (-)</option>
              <option value="json">JSON file</option>
              <option value="pkl">Pickle file</option>
              <option value="vector">Vector store</option>
              <option value="graph">Graph store</option>
              <option value="mongodb">MongoDB</option>
            </select>
            {(outputKind === 'json' || outputKind === 'pkl') && (
              <input
                style={{ marginTop: 8 }}
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
                placeholder={outputKind === 'json' ? 'out.json' : 'out.pkl'}
              />
            )}
            {outputKind === 'mongodb' && (
              <input style={{ marginTop: 8 }} value={mongoUri} onChange={(e) => setMongoUri(e.target.value)} placeholder="mongodb://host/db" />
            )}
            {outputError && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)', marginTop: 8 }}>{outputError}</div>
            )}
            <div style={{ marginTop: 12 }}>
              <div className="label-sm" style={{ marginBottom: 6 }}>
                Format
              </div>
              <select value={format} onChange={(e) => setFormat(e.target.value as 'auto' | 'json' | 'pkl')}>
                <option value="auto">auto</option>
                <option value="json">json</option>
                <option value="pkl">pkl</option>
              </select>
              <div style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 6 }}>
                Overrides the file extension.
              </div>
            </div>
          </div>

          {propertyNames.length > 0 && (
            <div>
              <div className="label" style={{ marginBottom: 8 }}>
                Parameters
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {propertyNames.map((name) => {
                  const schema = properties[name]
                  const value = params[name]
                  return (
                    <div key={name}>
                      <div className="label-sm" style={{ marginBottom: 6 }}>
                        {name}
                      </div>
                      {schema.type === 'boolean' ? (
                        <div style={{ display: 'flex', gap: 6 }}>
                          <button
                            type="button"
                            className={value === true ? 'btn btn-primary' : 'btn'}
                            style={{ flex: 1, justifyContent: 'center' }}
                            onClick={() => setParams((p) => ({ ...p, [name]: true }))}
                          >
                            TRUE
                          </button>
                          <button
                            type="button"
                            className={value === false ? 'btn btn-primary' : 'btn'}
                            style={{ flex: 1, justifyContent: 'center' }}
                            onClick={() => setParams((p) => ({ ...p, [name]: false }))}
                          >
                            FALSE
                          </button>
                        </div>
                      ) : schema.enum ? (
                        <select
                          value={typeof value === 'string' ? value : ''}
                          onChange={(e) => setParams((p) => ({ ...p, [name]: e.target.value }))}
                        >
                          <option value="" disabled>
                            select…
                          </option>
                          {schema.enum.map((choice) => (
                            <option key={choice} value={choice}>
                              {choice}
                            </option>
                          ))}
                        </select>
                      ) : schema.type === 'integer' || schema.type === 'number' ? (
                        <input
                          type="number"
                          value={typeof value === 'number' || typeof value === 'string' ? value : ''}
                          onChange={(e) => setParams((p) => ({ ...p, [name]: e.target.value === '' ? '' : Number(e.target.value) }))}
                        />
                      ) : (
                        <input
                          value={typeof value === 'string' ? value : value === undefined || value === null ? '' : String(value)}
                          onChange={(e) => setParams((p) => ({ ...p, [name]: e.target.value }))}
                        />
                      )}
                      {schema.description && (
                        <div style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 6 }}>
                          {schema.description}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
              {paramError && (
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)', marginTop: 8 }}>{paramError}</div>
              )}
            </div>
          )}
        </div>

        {error && (
          <div style={{ padding: '0 20px 12px', flexShrink: 0, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)' }}>
            {error}
          </div>
        )}

        <div
          style={{
            borderTop: '1px solid var(--c18)',
            padding: '14px 20px',
            flexShrink: 0,
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 8,
          }}
        >
          <button type="button" className="btn" onClick={onClose}>
            CANCEL
          </button>
          <button type="button" className="btn btn-primary" onClick={handleSubmit} disabled={submitDisabled}>
            {busy ? 'STARTING…' : 'RUN'}
          </button>
        </div>
      </div>

      <style>{`
        .run-dialog-x:hover { color: var(--cream); }
      `}</style>
    </div>
  )
}
