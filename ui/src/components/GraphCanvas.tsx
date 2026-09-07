import { useMemo, useState } from 'react'
import type { DragEvent } from 'react'
import type { AgentSpec, Pipeline } from '../types'
import { groupByStage } from '../yaml'
import NodeCard from './NodeCard'

type DragPos = { index: number; pos: 'above' | 'below' }

const BRANCH_W = 210
const BRANCH_GAP = 20

function Trunk({ height }: { height: number }) {
  return (
    <div style={{ position: 'relative', width: 1, height, background: 'var(--edge)' }}>
      {height >= 32 && (
        <div
          style={{
            position: 'absolute',
            top: height / 2 - 3,
            left: -2.5,
            width: 6,
            height: 6,
            background: 'var(--canvas)',
            border: '1px solid var(--edge)',
          }}
        />
      )}
    </div>
  )
}

/** The horizontal bar plus one stub per branch. `left/right: half a branch` lands the bar on the outer branch centres. */
function Bus({ count, dir }: { count: number; dir: 'split' | 'join' }) {
  return (
    <div style={{ display: 'flex', gap: BRANCH_GAP, position: 'relative', height: 14 }}>
      <div
        style={{
          position: 'absolute',
          top: dir === 'split' ? 0 : 13,
          left: BRANCH_W / 2,
          right: BRANCH_W / 2,
          height: 1,
          background: 'var(--edge)',
        }}
      />
      {Array.from({ length: count }, (_, i) => (
        <div key={i} style={{ width: BRANCH_W, display: 'flex', justifyContent: 'center' }}>
          <div style={{ width: 1, height: 14, background: 'var(--edge)' }} />
        </div>
      ))}
    </div>
  )
}

function Link({ fromCount, toCount }: { fromCount: number; toCount: number }) {
  const branched = fromCount > 1 || toCount > 1
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      {fromCount > 1 && <Bus count={fromCount} dir="join" />}
      <Trunk height={branched ? 20 : 40} />
      {toCount > 1 && <Bus count={toCount} dir="split" />}
    </div>
  )
}

export default function GraphCanvas({
  pipeline,
  agents,
  selectedUid,
  onSelect,
  onChange,
  onAddAgent,
}: {
  pipeline: Pipeline
  agents: AgentSpec[]
  selectedUid: string | null
  onSelect: (uid: string | null) => void
  onChange: (next: Pipeline) => void
  onAddAgent: (agentId: string, index: number) => void
}) {
  const specById = useMemo(() => new Map(agents.map((a) => [a.id, a])), [agents])
  const groups = useMemo(() => groupByStage(pipeline.steps), [pipeline.steps])
  const indexByUid = useMemo(() => {
    const m = new Map<string, number>()
    pipeline.steps.forEach((s, i) => m.set(s.uid, i))
    return m
  }, [pipeline.steps])
  const [dragOver, setDragOver] = useState<DragPos | null>(null)

  function groupStart(gi: number): number {
    let n = 0
    for (let i = 0; i < gi; i++) n += groups[i].steps.length
    return n
  }

  function targetIndex(gi: number, pos: 'above' | 'below'): number {
    return pos === 'above' ? groupStart(gi) : groupStart(gi) + groups[gi].steps.length
  }

  function dropOnGroup(e: DragEvent<HTMLDivElement>, gi: number) {
    e.preventDefault()
    e.stopPropagation()
    const pos = dragOver?.index === gi ? dragOver.pos : 'below'
    const target = targetIndex(gi, pos)

    const agentId = e.dataTransfer.getData('application/x-agent')
    if (agentId) {
      onAddAgent(agentId, target)
      setDragOver(null)
      return
    }

    const uid = e.dataTransfer.getData('application/x-step-uid')
    if (uid) {
      const steps = [...pipeline.steps]
      const from = steps.findIndex((s) => s.uid === uid)
      if (from !== -1) {
        const [moved] = steps.splice(from, 1)
        const to = from < target ? target - 1 : target
        steps.splice(to, 0, moved)
        onChange({ ...pipeline, steps })
      }
    }
    setDragOver(null)
  }

  function handleClear() {
    if (!pipeline.steps.length) return
    if (window.confirm('Clear the pipeline? This removes every step.')) onChange({ ...pipeline, steps: [] })
  }

  return (
    <div style={{ position: 'relative', flex: 1, minWidth: 0, height: '100%' }}>
      <div style={{ position: 'absolute', top: 16, left: 16, zIndex: 1 }}>
        <button className="btn btn-sm" onClick={handleClear}>
          CLEAR
        </button>
      </div>
      <div className="label" style={{ position: 'absolute', top: 16, right: 16, zIndex: 1 }}>
        {pipeline.steps.length} STEPS · {groups.length} STAGES
      </div>

      <div
        className="scroll"
        style={{
          position: 'absolute',
          inset: 0,
          background: 'var(--canvas)',
          backgroundImage: 'radial-gradient(var(--grid) 1px, transparent 1px)',
          backgroundSize: '22px 22px',
          padding: '56px 24px 40px',
        }}
        onClick={() => onSelect(null)}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          if (pipeline.steps.length) return
          const agentId = e.dataTransfer.getData('application/x-agent')
          if (agentId) onAddAgent(agentId, 0)
        }}
      >
        {pipeline.steps.length === 0 ? (
          <div className="empty" style={{ textAlign: 'center', marginTop: 100 }}>
            Empty pipeline. Click an agent in the library, or drag one here.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {groups.map((group, gi) => (
              <div key={group.steps[0].uid} style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                {gi > 0 && <Link fromCount={groups[gi - 1].steps.length} toCount={group.steps.length} />}

                {dragOver?.index === gi && dragOver.pos === 'above' && (
                  <div style={{ width: 260, height: 2, background: 'var(--cream)' }} />
                )}

                <div
                  onDragOver={(e) => {
                    e.preventDefault()
                    const rect = e.currentTarget.getBoundingClientRect()
                    const mid = rect.top + rect.height / 2
                    setDragOver({ index: gi, pos: e.clientY < mid ? 'above' : 'below' })
                  }}
                  onDrop={(e) => dropOnGroup(e, gi)}
                  style={group.stage ? { position: 'relative', border: '1px dashed var(--c40)', padding: 16 } : undefined}
                >
                  {group.stage && (
                    <div
                      style={{
                        position: 'absolute',
                        top: -7,
                        left: 12,
                        background: 'var(--canvas)',
                        padding: '0 7px',
                        fontFamily: 'var(--mono)',
                        fontSize: 8,
                        letterSpacing: '.2em',
                        color: 'var(--c62)',
                        textTransform: 'uppercase',
                      }}
                    >
                      STAGE · {group.stage} · CONCURRENT
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: group.steps.length > 1 ? BRANCH_GAP : 0 }}>
                    {group.steps.map((step) => (
                      <div
                        key={step.uid}
                        draggable
                        onDragStart={(e) => e.dataTransfer.setData('application/x-step-uid', step.uid)}
                        onClick={(e) => {
                          e.stopPropagation()
                          onSelect(step.uid)
                        }}
                      >
                        <NodeCard
                          step={step}
                          spec={specById.get(step.agent)}
                          index={indexByUid.get(step.uid) ?? 0}
                          selected={step.uid === selectedUid}
                          width={group.steps.length > 1 ? BRANCH_W : 260}
                          onDelete={() => onChange({ ...pipeline, steps: pipeline.steps.filter((s) => s.uid !== step.uid) })}
                        />
                      </div>
                    ))}
                  </div>
                </div>

                {dragOver?.index === gi && dragOver.pos === 'below' && (
                  <div style={{ width: 260, height: 2, background: 'var(--cream)' }} />
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
