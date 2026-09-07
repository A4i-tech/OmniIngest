import { useState } from 'react'
import type { AgentSpec, Pipeline } from '../types'
import { fromYaml, toYaml } from '../yaml'

type Span = { text: string; color?: string }

function colorLine(line: string): Span[] {
  const trimmed = line.trimStart()
  if (trimmed.startsWith('#')) return [{ text: line, color: 'var(--c50)' }]

  const kv = /^(\s*)(?:(-)(\s+))?([^:#\s][^:]*?):(\s*)(.*)$/.exec(line)
  if (kv) {
    const [, indent, dash, dsp, key, gap, rest] = kv
    const spans: Span[] = [{ text: indent }]
    if (dash) spans.push({ text: dash, color: 'var(--c50)' }, { text: dsp ?? '' })
    spans.push({ text: key, color: 'var(--cream)' }, { text: ':', color: 'var(--c50)' }, { text: gap })
    if (rest) {
      const hashIdx = rest.indexOf('#')
      if (hashIdx >= 0) spans.push({ text: rest.slice(0, hashIdx), color: 'var(--jq)' }, { text: rest.slice(hashIdx), color: 'var(--c50)' })
      else spans.push({ text: rest, color: 'var(--jq)' })
    }
    return spans
  }

  const item = /^(\s*)(-)(\s+)(.*)$/.exec(line)
  if (item) {
    const [, indent, dash, sp, rest] = item
    return [{ text: indent }, { text: dash, color: 'var(--c50)' }, { text: sp }, { text: rest, color: 'var(--jq)' }]
  }

  return [{ text: line }]
}

export default function YamlPane({
  pipeline,
  agents,
  onChange,
}: {
  pipeline: Pipeline
  agents: AgentSpec[]
  onChange: (next: Pipeline) => void
}) {
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState('')
  const [error, setError] = useState<string | undefined>(undefined)
  const [copied, setCopied] = useState(false)

  function toggleEdit() {
    if (!editing) {
      setText(toYaml(pipeline, agents))
      setError(undefined)
    }
    setEditing((v) => !v)
  }

  function handleTextChange(next: string) {
    setText(next)
    try {
      onChange(fromYaml(next))
      setError(undefined)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'This YAML could not be parsed. Fix the syntax and try again.')
    }
  }

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(toYaml(pipeline, agents))
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      setError('The browser blocked clipboard access. Use DOWNLOAD, or copy from the YAML pane after pressing EDIT.')
    }
  }

  function handleDownload() {
    const blob = new Blob([toYaml(pipeline, agents)], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${pipeline.pipeline_id || 'pipeline'}.yaml`
    a.click()
    URL.revokeObjectURL(url)
  }

  const linkStyle = {
    border: 'none',
    background: 'none',
    padding: 0,
    height: 'auto',
    fontFamily: 'var(--mono)',
    fontSize: 9,
    letterSpacing: '.16em',
    color: 'var(--c62)',
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div
        style={{
          height: 38,
          borderBottom: '1px solid var(--c18)',
          padding: '0 14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}
      >
        <div style={{ fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '.16em', color: 'var(--c50)' }}>
          {pipeline.pipeline_id || 'untitled'}.yaml
        </div>
        <div style={{ display: 'flex', gap: 14 }}>
          <button style={linkStyle} onClick={toggleEdit}>
            {editing ? 'PREVIEW' : 'EDIT'}
          </button>
          <button style={linkStyle} onClick={handleCopy}>
            {copied ? 'COPIED' : 'COPY'}
          </button>
          <button style={linkStyle} onClick={handleDownload}>
            DOWNLOAD
          </button>
        </div>
      </div>

      {editing ? (
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {error && <div style={{ padding: '6px 14px', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--failed)' }}>{error}</div>}
          <textarea
            value={text}
            onChange={(e) => handleTextChange(e.target.value)}
            style={{ flex: 1, height: '100%', resize: 'none', fontSize: 10, lineHeight: 1.62 }}
          />
        </div>
      ) : (
        <div
          className="scroll"
          style={{ flex: 1, minHeight: 0, fontFamily: 'var(--mono)', fontSize: 10, lineHeight: 1.62, padding: '12px 18px', whiteSpace: 'pre-wrap' }}
        >
          {toYaml(pipeline, agents)
            .split('\n')
            .map((line, i) => (
              <div key={i}>
                {line === '' ? ' ' : colorLine(line).map((s, j) => <span key={j} style={{ color: s.color }}>{s.text}</span>)}
              </div>
            ))}
        </div>
      )}
    </div>
  )
}
