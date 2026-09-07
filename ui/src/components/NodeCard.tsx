import { useState } from 'react'
import type { AgentSpec, StepNode } from '../types'

function SquareIcon() {
  return <div className="sq" style={{ background: 'var(--cream)' }} />
}

function ComponentIcon() {
  return (
    <svg width={9} height={9} viewBox="0 0 9 9" style={{ flexShrink: 0 }}>
      <rect x={1} y={1} width={7} height={7} fill="none" stroke="var(--jq)" strokeWidth={1} />
      <line x1={1} y1={1} x2={8} y2={8} stroke="var(--jq)" strokeWidth={1} />
      <line x1={8} y1={1} x2={1} y2={8} stroke="var(--jq)" strokeWidth={1} />
    </svg>
  )
}

function TriangleIcon() {
  return (
    <svg width={9} height={9} viewBox="0 0 9 9" style={{ flexShrink: 0 }}>
      <polygon points="4.5,1 8,8 1,8" fill="none" stroke="var(--failed)" strokeWidth={1} />
    </svg>
  )
}

function XIcon() {
  return (
    <svg width={9} height={9} viewBox="0 0 9 9">
      <line x1={1} y1={1} x2={8} y2={8} stroke="currentColor" strokeWidth={1.2} />
      <line x1={8} y1={1} x2={1} y2={8} stroke="currentColor" strokeWidth={1.2} />
    </svg>
  )
}

function fmtValue(v: unknown): string {
  if (typeof v === 'string') return v
  if (v === null || v === undefined) return ''
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

export default function NodeCard({
  step,
  spec,
  index,
  selected,
  width,
  onDelete,
}: {
  step: StepNode
  spec?: AgentSpec
  index: number
  selected: boolean
  width: number
  onDelete: () => void
}) {
  const [hovered, setHovered] = useState(false)
  const [delHovered, setDelHovered] = useState(false)

  const configEntries = Object.entries(step.config)
  const modes = [...new Set(step.validation.map((v) => v.mode.toUpperCase()))].join(', ')

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: 'relative',
        width,
        background: 'var(--blue)',
        border: selected ? '2px solid var(--accent)' : '1px solid var(--c28)',
        padding: selected ? '10px 12px' : '11px 13px',
      }}
    >
      <button
        onMouseEnter={() => setDelHovered(true)}
        onMouseLeave={() => setDelHovered(false)}
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        style={{
          position: 'absolute',
          top: 6,
          right: 6,
          border: 'none',
          background: 'none',
          padding: 0,
          height: 'auto',
          cursor: 'pointer',
          opacity: hovered ? 1 : 0,
          color: delHovered ? 'var(--failed)' : 'var(--c50)',
        }}
      >
        <XIcon />
      </button>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, minWidth: 0 }}>
          {spec?.kind === 'component' ? <ComponentIcon /> : <SquareIcon />}
          <span
            style={{
              fontFamily: 'var(--mono)',
              fontWeight: 500,
              fontSize: 12,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {step.agent}
          </span>
        </div>
        <span style={{ fontFamily: 'var(--serif)', fontSize: 15, color: 'var(--c40)', flexShrink: 0 }}>
          {String(index + 1).padStart(2, '0')}
        </span>
      </div>

      <div style={{ borderTop: '1px solid var(--c18)', marginTop: 8, paddingTop: 8 }}>
        {spec?.kind === 'component' ? (
          <>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--jq)' }}>
              COMPONENT · {spec.steps?.length ?? 0} {spec.steps?.length === 1 ? 'STEP' : 'STEPS'}
            </div>
            <div style={{ fontFamily: 'var(--sans)', fontSize: 11, color: 'var(--c62)', marginTop: 4 }}>{spec.path}</div>
          </>
        ) : configEntries.length === 0 ? (
          <div style={{ fontFamily: 'var(--sans)', fontSize: 11, color: 'var(--c62)' }}>{spec?.description}</div>
        ) : (
          <>
            {configEntries.slice(0, 3).map(([k, v]) => {
              const field = spec?.fields.find((f) => f.name === k)
              return (
                <div
                  key={k}
                  style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontFamily: 'var(--mono)', fontSize: 10, padding: '2px 0' }}
                >
                  <span style={{ color: 'var(--c50)', flexShrink: 0 }}>{k}</span>
                  <span
                    style={{
                      color: field?.type === 'jq' ? 'var(--jq)' : undefined,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {fmtValue(v)}
                  </span>
                </div>
              )
            })}
            {configEntries.length > 3 && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--c50)' }}>{configEntries.length - 3} more</div>
            )}
          </>
        )}

        {step.validation.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
            <TriangleIcon />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--failed)' }}>
              {step.validation.length} VALIDATION · {modes}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
