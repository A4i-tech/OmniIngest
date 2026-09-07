import type { ValidationCheck, ValidationMode } from '../types'

interface ValidationEditorProps {
  checks: ValidationCheck[]
  onChange: (next: ValidationCheck[]) => void
}

const MODE_MEANING: Record<ValidationMode, string> = {
  condition: 'Skips this step when the expression is false.',
  before: 'Runs before the step.',
  after: 'Runs after a successful step; can inspect .result.',
  plain: 'Runs before the step; if it passes, stops the rest of the chain for this step.',
}

const MODES: ValidationMode[] = ['condition', 'before', 'plain', 'after']

function XIcon() {
  return (
    <svg width="9" height="9" viewBox="0 0 9 9">
      <line x1="1" y1="1" x2="8" y2="8" stroke="currentColor" strokeWidth="1.2" />
      <line x1="8" y1="1" x2="1" y2="8" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  )
}

export default function ValidationEditor({ checks, onChange }: ValidationEditorProps) {
  function update(i: number, patch: Partial<ValidationCheck>) {
    onChange(checks.map((c, idx) => (idx === i ? { ...c, ...patch } : c)))
  }

  function remove(i: number) {
    onChange(checks.filter((_, idx) => idx !== i))
  }

  function add() {
    onChange([...checks, { mode: 'before', expr: '', message: '' }])
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <span className="label">VALIDATION</span>
        <button type="button" className="btn btn-sm" onClick={add}>
          + CHECK
        </button>
      </div>

      {checks.length === 0 && (
        <div className="empty">No checks. This step always runs and is never verified.</div>
      )}

      {checks.map((check, i) => {
        const exprEmpty = check.expr.trim() === ''
        const usesParam = check.expr.includes('${')
        return (
          <div
            key={i}
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
              paddingTop: i === 0 ? 0 : 12,
              marginTop: i === 0 ? 0 : 12,
              borderTop: i === 0 ? undefined : '1px solid var(--c18)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <select
                style={{ width: 120, flexShrink: 0 }}
                value={check.mode}
                onChange={(e) => update(i, { mode: e.target.value as ValidationMode })}
              >
                {MODES.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <span style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', flex: 1 }}>
                {MODE_MEANING[check.mode]}
              </span>
              <button
                type="button"
                onClick={() => remove(i)}
                style={{
                  border: 'none',
                  background: 'transparent',
                  padding: 0,
                  color: 'var(--c50)',
                  display: 'flex',
                  flexShrink: 0,
                }}
                className="validation-x"
                aria-label="Delete check"
              >
                <XIcon />
              </button>
            </div>

            <div>
              <div className="label-sm" style={{ marginBottom: 6 }}>
                EXPR · JQ
              </div>
              <input
                value={check.expr}
                onChange={(e) => update(i, { expr: e.target.value })}
                placeholder=".metadata.score >= 0.6"
                style={{ color: 'var(--jq)', borderColor: exprEmpty ? 'var(--failed)' : undefined }}
              />
              {exprEmpty && (
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)', marginTop: 6 }}>
                  {check.mode === 'condition'
                    ? 'An empty expression is always false. This step will never run.'
                    : 'An empty expression fails this check every time.'}
                </div>
              )}
              {usesParam && (
                <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--jq)', marginTop: 6 }}>
                  USES A PIPELINE PARAMETER
                </div>
              )}
            </div>

            <div>
              <div className="label-sm" style={{ marginBottom: 6 }}>
                MESSAGE
              </div>
              <input
                value={check.message}
                onChange={(e) => update(i, { message: e.target.value })}
                placeholder="Explain what failed and what to do next."
              />
              {check.message.trim() === '' && (
                <div style={{ fontFamily: 'var(--sans)', fontSize: 10, color: 'var(--c62)', marginTop: 6 }}>
                  Without a message the run reports only the expression.
                </div>
              )}
            </div>
          </div>
        )
      })}

      <style>{`
        .validation-x:hover { color: var(--failed); }
      `}</style>
    </div>
  )
}
