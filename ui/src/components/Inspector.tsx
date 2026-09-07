import { useEffect, useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import type { AgentSpec, FieldSpec, StepNode } from '../types'
import { pruneConfig } from '../yaml'
import ValidationEditor from './ValidationEditor'

const isBlank = (v: unknown) =>
  v === undefined ||
  v === null ||
  v === '' ||
  (Array.isArray(v) && v.length === 0) ||
  (typeof v === 'object' && !Array.isArray(v) && Object.keys(v as object).length === 0)

function isShort(type: FieldSpec['type'], value: unknown): boolean {
  if (type === 'integer' || type === 'number' || type === 'boolean' || type === 'enum') return true
  if (type === 'text' || type === 'jq' || type === 'array' || type === 'object') return false
  return typeof value !== 'string' || value.length < 24
}

export default function Inspector({
  step,
  spec,
  index,
  onChange,
  onDelete,
  groupedWithPrevious,
  onToggleGroup,
}: {
  step: StepNode | null
  spec?: AgentSpec
  index: number
  onChange: (next: StepNode) => void
  onDelete: () => void
  /** null when this is the first step, so there is nothing to run alongside. */
  groupedWithPrevious: boolean | null
  onToggleGroup: () => void
}) {
  const [rawText, setRawText] = useState<Record<string, string>>({})
  const [jsonError, setJsonError] = useState<Record<string, string | undefined>>({})

  useEffect(() => {
    setRawText({})
    setJsonError({})
  }, [step?.uid])

  if (!step) {
    return (
      <div style={{ height: '100%', padding: 18 }}>
        <div className="empty">Select a step to edit its configuration.</div>
      </div>
    )
  }

  function setField(name: string, value: unknown) {
    if (!step) return
    const config = pruneConfig({ ...step.config, [name]: value }, spec)
    onChange({ ...step, config })
  }

  const fields = spec?.fields ?? []

  return (
    <div className="scroll" style={{ height: '100%', padding: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div className="label">STEP {String(index + 1).padStart(2, '0')}</div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '.16em', color: 'var(--c50)' }}>{spec?.cls}</div>
      </div>
      <div style={{ fontFamily: 'var(--serif)', fontSize: 30, lineHeight: 1, marginTop: 6, marginBottom: 16 }}>{step.agent}</div>

      <div style={{ marginBottom: 12 }}>
        <div className="label-sm" style={{ marginBottom: 4 }}>
          STAGE
        </div>
        <input
          value={step.stage ?? ''}
          placeholder="runs alone"
          onChange={(e) => onChange({ ...step, stage: e.target.value || undefined })}
        />
        {groupedWithPrevious !== null && (
          <button className="btn btn-sm" style={{ marginTop: 6, width: '100%', justifyContent: 'center' }} onClick={onToggleGroup}>
            {groupedWithPrevious ? 'SPLIT FROM PREVIOUS STEP' : 'RUN WITH PREVIOUS STEP'}
          </button>
        )}
        <div style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 5, lineHeight: 1.45 }}>
          Steps that share a stage run concurrently. They must sit next to each other.
        </div>
      </div>
      <div style={{ marginBottom: 16 }}>
        <div className="label-sm" style={{ marginBottom: 4 }}>
          DESCRIPTION
        </div>
        <input value={step.description ?? ''} onChange={(e) => onChange({ ...step, description: e.target.value || undefined })} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 11 }}>
        {fields.map((field) => {
          const raw = step.config[field.name]
          const usesDefault = !field.required && (raw === undefined || (field.default !== undefined && JSON.stringify(raw) === JSON.stringify(field.default)))
          const displayValue = usesDefault ? undefined : raw
          const short = isShort(field.type, displayValue)
          const spanBoth = field.type === 'text' || field.type === 'jq' || field.type === 'array' || field.type === 'object' || !short
          const requiredEmpty = field.required && isBlank(raw)

          const label = (
            <div className="label-sm" style={{ marginBottom: 4 }}>
              {field.name.toUpperCase()}
              {field.required && <span style={{ color: 'var(--failed)' }}> *</span>}
            </div>
          )
          const desc = field.description ? (
            <div style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 4 }}>{field.description}</div>
          ) : null
          const borderStyle: CSSProperties = requiredEmpty ? { borderColor: 'var(--failed)' } : {}

          let control: ReactNode

          if (field.type === 'text') {
            control = (
              <textarea
                rows={4}
                placeholder={usesDefault ? 'default' : undefined}
                value={typeof displayValue === 'string' ? displayValue : ''}
                onChange={(e) => setField(field.name, e.target.value)}
                style={borderStyle}
              />
            )
          } else if (field.type === 'jq') {
            control = (
              <input
                placeholder={usesDefault ? 'default' : undefined}
                value={typeof displayValue === 'string' ? displayValue : ''}
                onChange={(e) => setField(field.name, e.target.value)}
                style={{ color: 'var(--jq)', ...borderStyle }}
              />
            )
          } else if (field.type === 'integer' || field.type === 'number') {
            control = (
              <input
                type="number"
                step={field.type === 'integer' ? 1 : undefined}
                placeholder={usesDefault ? 'default' : undefined}
                value={typeof displayValue === 'number' ? displayValue : ''}
                onChange={(e) => setField(field.name, e.target.value === '' ? '' : Number(e.target.value))}
                style={borderStyle}
              />
            )
          } else if (field.type === 'boolean') {
            const effective = typeof raw === 'boolean' ? raw : Boolean(field.default)
            control = (
              <div style={{ display: 'flex', gap: 6 }}>
                <button className={effective ? 'btn btn-primary' : 'btn'} style={{ flex: 1 }} onClick={() => setField(field.name, true)}>
                  TRUE
                </button>
                <button className={!effective ? 'btn btn-primary' : 'btn'} style={{ flex: 1 }} onClick={() => setField(field.name, false)}>
                  FALSE
                </button>
              </div>
            )
          } else if (field.type === 'enum') {
            const effective = typeof raw === 'string' ? raw : (field.default as string | undefined) ?? ''
            control = (
              <select value={effective} onChange={(e) => setField(field.name, e.target.value)} style={borderStyle}>
                {/* Without this, an unset or retired value renders as the first option while config holds something else. */}
                {!(field.choices ?? []).includes(effective) && <option value={effective}>{effective === '' ? '— not set —' : effective}</option>}
                {(field.choices ?? []).map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            )
          } else if (field.type === 'array' || field.type === 'object') {
            const fallback = field.type === 'array' ? [] : {}
            const text = rawText[field.name] ?? (usesDefault ? '' : JSON.stringify(displayValue ?? fallback, null, 2))
            control = (
              <>
                <textarea
                  rows={4}
                  placeholder={usesDefault ? 'default' : undefined}
                  value={text}
                  onChange={(e) => setRawText((r) => ({ ...r, [field.name]: e.target.value }))}
                  onBlur={(e) => {
                    const t = e.target.value
                    if (t.trim() === '') {
                      setField(field.name, fallback)
                      setJsonError((r) => ({ ...r, [field.name]: undefined }))
                      setRawText((r) => {
                        const next = { ...r }
                        delete next[field.name]
                        return next
                      })
                      return
                    }
                    try {
                      const parsed = JSON.parse(t)
                      setField(field.name, parsed)
                      setJsonError((r) => ({ ...r, [field.name]: undefined }))
                      setRawText((r) => {
                        const next = { ...r }
                        delete next[field.name]
                        return next
                      })
                    } catch {
                      setJsonError((r) => ({ ...r, [field.name]: 'Not valid JSON — fix the highlighted field before running.' }))
                    }
                  }}
                  style={jsonError[field.name] ? { borderColor: 'var(--failed)' } : borderStyle}
                />
                {jsonError[field.name] && (
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)', marginTop: 4 }}>{jsonError[field.name]}</div>
                )}
              </>
            )
          } else {
            control = (
              <input
                placeholder={usesDefault ? 'default' : undefined}
                value={typeof displayValue === 'string' ? displayValue : ''}
                onChange={(e) => setField(field.name, e.target.value)}
                style={borderStyle}
              />
            )
          }

          return (
            <div key={field.name} style={{ gridColumn: spanBoth ? '1 / -1' : undefined }}>
              {label}
              {control}
              {desc}
            </div>
          )
        })}
      </div>

      <div style={{ borderTop: '1px solid var(--c18)', marginTop: 18, paddingTop: 14 }}>
        <ValidationEditor checks={step.validation} onChange={(validation) => onChange({ ...step, validation })} />
      </div>

      <button className="btn btn-danger btn-sm" style={{ marginTop: 18 }} onClick={onDelete}>
        DELETE STEP
      </button>
    </div>
  )
}
