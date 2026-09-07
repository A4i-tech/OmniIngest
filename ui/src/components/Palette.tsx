import { useMemo, useState } from 'react'
import type { AgentSpec } from '../types'

const GROUP_ORDER = ['modality', 'chunking', 'document', 'enrichment', 'indexing', 'governance']

function DotsIcon() {
  return (
    <svg width={7} height={11} viewBox="0 0 7 11" style={{ flexShrink: 0 }}>
      {[1, 5, 9].flatMap((y) => [1, 5].map((x) => <circle key={`${x}-${y}`} cx={x} cy={y} r={1} fill="var(--c50)" />))}
    </svg>
  )
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

function groupAgents(agents: AgentSpec[]): Array<[string, AgentSpec[]]> {
  const map = new Map<string, AgentSpec[]>()
  for (const a of agents) {
    const list = map.get(a.group)
    if (list) list.push(a)
    else map.set(a.group, [a])
  }
  const rest = [...map.keys()].filter((g) => g !== 'components' && !GROUP_ORDER.includes(g))
  const order = [...GROUP_ORDER.filter((g) => map.has(g)), ...rest, ...(map.has('components') ? ['components'] : [])]
  return order.map((g) => [g, map.get(g) as AgentSpec[]])
}

export default function Palette({ agents, onAdd }: { agents: AgentSpec[]; onAdd: (spec: AgentSpec) => void }) {
  const [filter, setFilter] = useState('')

  const groups = useMemo(() => {
    const q = filter.trim().toLowerCase()
    const filtered = q ? agents.filter((a) => a.id.toLowerCase().includes(q)) : agents
    return groupAgents(filtered)
  }, [agents, filter])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <style>{'.palette-row:hover { background: var(--hover); }'}</style>
      <div style={{ padding: '18px 18px 10px' }}>
        <div className="label">AGENT LIBRARY</div>
      </div>
      <div style={{ padding: '0 18px 12px' }}>
        <input placeholder="filter agents" value={filter} onChange={(e) => setFilter(e.target.value)} />
      </div>
      <div className="scroll" style={{ flex: 1 }}>
        {groups.map(([group, list]) => (
          <div key={group}>
            <div
              className="label-sm"
              style={{ padding: '10px 18px 4px', textTransform: group === 'components' ? 'none' : undefined }}
            >
              {group === 'components' ? 'COMPONENTS · profiles/' : group.toUpperCase()}
            </div>
            {list.map((a) => (
              <div
                key={a.id}
                className="palette-row"
                draggable
                title={a.description}
                onDragStart={(e) => e.dataTransfer.setData('application/x-agent', a.id)}
                onClick={() => onAdd(a)}
                style={{ height: 26, padding: '0 18px', display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
              >
                {a.kind === 'component' ? <ComponentIcon /> : <DotsIcon />}
                <span
                  style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 11,
                    color: a.kind === 'component' ? 'var(--jq)' : undefined,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {a.id}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
