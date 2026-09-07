import { useRef, useState } from 'react'
import type { ChangeEvent, MouseEvent } from 'react'
import type { SavedPipeline } from '../types'

interface LibraryProps {
  items: SavedPipeline[]
  loading: boolean
  error: string | null
  busy: string | null
  onOpen: (name: string) => void
  onDelete: (name: string) => void
  onImport: (name: string, yamlText: string) => void
  onNew: () => void
}

type SourceFilter = 'ALL' | 'PROFILES/' | 'TESTS/' | 'LOCAL'

const FILTERS: SourceFilter[] = ['ALL', 'PROFILES/', 'TESTS/', 'LOCAL']

// Agents that touch the knowledge graph get the jq accent, per SPEC.
const KG_AGENTS = new Set(['enrichment', 'quality', 'kg_builder', 'learning_outcome', 'faq_generation', 'keyword_indexing'])

function matchesFilter(item: SavedPipeline, filter: SourceFilter): boolean {
  if (filter === 'ALL') return true
  if (filter === 'PROFILES/') return item.source === 'profiles'
  if (filter === 'TESTS/') return item.source === 'tests'
  return item.source === 'local'
}

function sourceLabel(source: SavedPipeline['source']): string {
  if (source === 'profiles') return 'profiles/'
  if (source === 'tests') return 'tests/integration/'
  return 'DRAFT · LOCAL'
}

function relativeTime(modified: number | undefined): string {
  if (modified === undefined) return ''
  const minutes = Math.floor((Date.now() - modified) / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

function AgentChips({ agents }: { agents: string[] }) {
  const shown = agents.slice(0, 4)
  const hidden = agents.length - shown.length
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
      {shown.map((agent, i) => (
        <span
          key={`${agent}-${i}`}
          style={{
            border: '1px solid var(--c28)',
            padding: '2px 7px',
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: KG_AGENTS.has(agent) ? 'var(--jq)' : undefined,
          }}
        >
          {agent}
        </span>
      ))}
      {hidden > 0 && (
        <span style={{ border: '1px solid var(--c28)', padding: '2px 7px', fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--c50)' }}>
          +{hidden}
        </span>
      )}
    </div>
  )
}

interface CardProps {
  item: SavedPipeline
  busy: boolean
  onOpen: (name: string) => void
  onDelete: (e: MouseEvent, name: string) => void
}

function Card({ item, busy, onOpen, onDelete }: CardProps) {
  const [hover, setHover] = useState(false)
  const isLocal = item.source === 'local'
  const borderColor = busy ? 'var(--running)' : hover ? 'var(--cream)' : 'var(--c40)'

  return (
    <div
      className="lib-card"
      onClick={() => onOpen(item.name)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: 'relative',
        border: `1px ${isLocal ? 'dashed' : 'solid'} ${borderColor}`,
        padding: '18px 18px 16px',
        height: 208,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        cursor: 'pointer',
      }}
    >
      {isLocal && hover && !busy && (
        <button
          className="lib-del"
          onClick={(e) => onDelete(e, item.name)}
          aria-label={`Delete ${item.name}`}
          style={{ position: 'absolute', top: 12, right: 12, width: 9, height: 9, padding: 0, border: 'none', background: 'transparent' }}
        >
          <svg width={9} height={9} viewBox="0 0 9 9">
            <line className="lib-del-line" x1="1" y1="1" x2="8" y2="8" strokeWidth="1" />
            <line className="lib-del-line" x1="8" y1="1" x2="1" y2="8" strokeWidth="1" />
          </svg>
        </button>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div className="label-sm" style={busy ? { color: 'var(--running)' } : undefined}>
          {busy ? 'WORKING…' : sourceLabel(item.source)}
        </div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--c62)' }}>{relativeTime(item.modified)}</div>
      </div>
      <div style={{ fontFamily: 'var(--serif)', fontSize: 25, lineHeight: 1.05 }}>{item.name}</div>
      <div
        style={{
          fontFamily: 'var(--sans)',
          fontSize: 12,
          lineHeight: 1.45,
          color: item.description ? 'var(--c62)' : 'var(--c50)',
          flexGrow: 1,
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}
      >
        {item.description || 'No description in this YAML.'}
      </div>
      <AgentChips agents={item.agents} />
    </div>
  )
}

export default function Library({ items, loading, error, busy, onOpen, onDelete, onImport, onNew }: LibraryProps) {
  const [filter, setFilter] = useState<SourceFilter>('ALL')
  const [search, setSearch] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  const filtered = items.filter((item) => {
    if (!matchesFilter(item, filter)) return false
    const q = search.trim().toLowerCase()
    if (!q) return true
    return item.name.toLowerCase().includes(q) || (item.description ?? '').toLowerCase().includes(q)
  })

  async function handleFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const text = await file.text()
    onImport(file.name.replace(/\.ya?ml$/i, ''), text)
    if (fileInput.current) fileInput.current.value = ''
  }

  function handleDelete(e: MouseEvent, name: string) {
    e.stopPropagation()
    if (window.confirm(`Delete pipeline "${name}"? This cannot be undone.`)) onDelete(name)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <style>{`
        .lib-del-line { stroke: var(--c50); }
        .lib-del:hover .lib-del-line { stroke: var(--failed); }
      `}</style>

      <div style={{ height: 44, borderBottom: '1px solid var(--c18)', padding: '0 32px', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 8 }}>
        <input ref={fileInput} type="file" accept=".yaml,.yml" style={{ display: 'none' }} onChange={handleFile} />
        <button className="btn" onClick={() => fileInput.current?.click()}>IMPORT YAML</button>
        <button className="btn btn-primary" onClick={onNew}>+ NEW PIPELINE</button>
      </div>

      <div style={{ padding: '30px 32px 22px', borderBottom: '1px solid var(--c18)', display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between' }}>
        <div>
          <div className="label">SAVED PIPELINES · {filtered.length}</div>
          <div style={{ fontFamily: 'var(--serif)', fontSize: 58, lineHeight: 0.92 }}>Library</div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 10 }}>
          <div style={{ display: 'flex', gap: 24 }}>
            {FILTERS.map((f) => (
              <div
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  cursor: 'pointer',
                  fontFamily: 'var(--mono)',
                  fontSize: 10,
                  letterSpacing: '.14em',
                  paddingBottom: 3,
                  color: filter === f ? 'var(--cream)' : 'var(--c50)',
                  borderBottom: `1px solid ${filter === f ? 'var(--cream)' : 'transparent'}`,
                }}
              >
                {f}
              </div>
            ))}
          </div>
          <input type="text" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="search pipelines" style={{ width: 220 }} />
        </div>
      </div>

      <div className="scroll" style={{ flexGrow: 1, padding: '26px 32px', display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 20, alignContent: 'start' }}>
        {loading ? (
          <div className="empty" style={{ gridColumn: '1 / -1' }}>Loading pipelines from the server…</div>
        ) : error ? (
          <div style={{ gridColumn: '1 / -1' }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--failed)' }}>{error}</div>
            <div style={{ fontFamily: 'var(--sans)', fontSize: 12, color: 'var(--c62)', marginTop: 6 }}>Check that the server is running, then reload.</div>
          </div>
        ) : items.length === 0 ? (
          <div className="empty" style={{ gridColumn: '1 / -1' }}>No pipelines found. Import a YAML file, or start a new pipeline.</div>
        ) : filtered.length === 0 ? (
          <div className="empty" style={{ gridColumn: '1 / -1' }}>No pipeline matches this filter.</div>
        ) : (
          filtered.map((item) => <Card key={item.name} item={item} busy={busy === item.name} onOpen={onOpen} onDelete={handleDelete} />)
        )}
      </div>
    </div>
  )
}
