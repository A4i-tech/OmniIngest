import { useMemo, useState } from 'react'
import type { AgentSpec, Pipeline, RunRequest, StepNode } from '../types'
import { newStep, validatePipeline } from '../yaml'
import Palette from '../components/Palette'
import GraphCanvas from '../components/GraphCanvas'
import Inspector from '../components/Inspector'
import YamlPane from '../components/YamlPane'
import PipelineSettings from '../components/PipelineSettings'
import RunDialog from '../components/RunDialog'

interface BuilderProps {
  agents: AgentSpec[]
  pipeline: Pipeline
  onChange: (next: Pipeline) => void
  onRun: (req: RunRequest) => void
  onSave: (name: string) => void
  running: boolean
  saving: boolean
  runError: string | null
}

function RunIcon() {
  return (
    <svg width={9} height={10} viewBox="0 0 9 10">
      <polygon points="0,0 9,5 0,10" fill="currentColor" />
    </svg>
  )
}

export default function Builder({ agents, pipeline, onChange, onRun, onSave, running, saving, runError }: BuilderProps) {
  const [selectedUid, setSelectedUid] = useState<string | null>(null)
  const [showRun, setShowRun] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  const specById = useMemo(() => new Map(agents.map((a) => [a.id, a])), [agents])
  const errors = useMemo(() => validatePipeline(pipeline, agents), [pipeline, agents])

  const selectedIndex = pipeline.steps.findIndex((s) => s.uid === selectedUid)
  const selectedStep = selectedIndex === -1 ? null : pipeline.steps[selectedIndex]

  function addAgent(spec: AgentSpec) {
    onChange({ ...pipeline, steps: [...pipeline.steps, newStep(spec)] })
  }

  function addAgentAt(agentId: string, index: number) {
    const spec = specById.get(agentId)
    if (!spec) return
    const steps = [...pipeline.steps]
    steps.splice(index, 0, newStep(spec))
    onChange({ ...pipeline, steps })
  }

  function updateStep(next: StepNode) {
    onChange({ ...pipeline, steps: pipeline.steps.map((s) => (s.uid === next.uid ? next : s)) })
  }

  function deleteStep(uid: string) {
    onChange({ ...pipeline, steps: pipeline.steps.filter((s) => s.uid !== uid) })
    if (selectedUid === uid) setSelectedUid(null)
  }

  const prev = selectedIndex > 0 ? pipeline.steps[selectedIndex - 1] : null
  const groupedWithPrevious = prev === null || !selectedStep ? null : Boolean(selectedStep.stage) && selectedStep.stage === prev.stage

  function freshStageName(): string {
    const taken = new Set(pipeline.steps.map((s) => s.stage).filter(Boolean))
    if (!taken.has('fanout')) return 'fanout'
    for (let n = 2; ; n++) if (!taken.has(`fanout-${n}`)) return `fanout-${n}`
  }

  // Grouping is always "join the step above": the runner requires a stage to be contiguous.
  function toggleGroup() {
    if (!selectedStep || !prev) return
    const stage = groupedWithPrevious ? undefined : prev.stage || freshStageName()
    const steps = pipeline.steps.map((s) => {
      if (s.uid === selectedStep.uid) return { ...s, stage }
      if (!groupedWithPrevious && s.uid === prev.uid) return { ...s, stage }
      return s
    })
    onChange({ ...pipeline, steps })
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{ height: 44, borderBottom: '1px solid var(--c18)', padding: '0 20px', display: 'flex', alignItems: 'center', gap: 10 }}>
        <input
          style={{ width: 260 }}
          placeholder="pipeline_id"
          value={pipeline.pipeline_id}
          onChange={(e) => onChange({ ...pipeline, pipeline_id: e.target.value })}
        />
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => setShowSettings(true)}>
          SETTINGS
        </button>
        <button className="btn" disabled={saving || !pipeline.pipeline_id} onClick={() => onSave(pipeline.pipeline_id)}>
          SAVE
        </button>
        <button className="btn btn-primary" disabled={running || errors.length > 0} onClick={() => setShowRun(true)}>
          <RunIcon /> RUN
        </button>
      </div>

      {pipeline.steps.length > 0 && errors.length > 0 && (
        <div style={{ padding: '8px 20px', borderBottom: '1px solid var(--c18)' }}>
          {errors.map((e, i) => (
            <div key={i} style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)' }}>
              {e}
            </div>
          ))}
        </div>
      )}

      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <div style={{ width: 248, borderRight: '1px solid var(--c22)' }}>
          <Palette agents={agents} onAdd={addAgent} />
        </div>

        <GraphCanvas
          pipeline={pipeline}
          agents={agents}
          selectedUid={selectedUid}
          onSelect={setSelectedUid}
          onChange={onChange}
          onAddAgent={addAgentAt}
        />

        <div style={{ width: 400, borderLeft: '1px solid var(--c22)', display: 'flex', flexDirection: 'column' }}>
          <div style={{ height: 384, borderBottom: '1px solid var(--c22)', flexShrink: 0 }}>
            <Inspector
              step={selectedStep}
              spec={selectedStep ? specById.get(selectedStep.agent) : undefined}
              index={selectedIndex === -1 ? 0 : selectedIndex}
              onChange={updateStep}
              onDelete={() => selectedStep && deleteStep(selectedStep.uid)}
              groupedWithPrevious={groupedWithPrevious}
              onToggleGroup={toggleGroup}
            />
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <YamlPane pipeline={pipeline} agents={agents} onChange={onChange} />
          </div>
        </div>

        {showSettings && <PipelineSettings pipeline={pipeline} onChange={onChange} onClose={() => setShowSettings(false)} />}
      </div>

      {showRun && (
        <RunDialog
          pipeline={pipeline}
          busy={running}
          error={runError}
          onClose={() => setShowRun(false)}
          onSubmit={(req) => {
            setShowRun(false)
            onRun(req)
          }}
        />
      )}
    </div>
  )
}
