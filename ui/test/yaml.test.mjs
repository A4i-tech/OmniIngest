import { readFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import assert from 'node:assert/strict'
import ts from 'typescript'
import yaml from 'js-yaml'

const yamlTsPath = fileURLToPath(new URL('../src/yaml.ts', import.meta.url))
const jsYamlEntry = fileURLToPath(new URL('../node_modules/js-yaml/index.js', import.meta.url))
if (!existsSync(jsYamlEntry)) throw new Error(`js-yaml entry point not found at ${jsYamlEntry}. Check node_modules/js-yaml/package.json main field.`)
const jsYamlUrl = new URL('../node_modules/js-yaml/index.js', import.meta.url).href

const src = readFileSync(yamlTsPath, 'utf8')
const patched = src.replace("from 'js-yaml'", `from '${jsYamlUrl}'`)
const js = ts.transpileModule(patched, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
const mod = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'))
const { fromYaml, toYaml, groupByStage, pruneConfig, validatePipeline } = mod

function stripUid(value) {
  if (Array.isArray(value)) return value.map(stripUid)
  if (value && typeof value === 'object') {
    const out = {}
    for (const [k, v] of Object.entries(value)) {
      if (k !== 'uid') out[k] = stripUid(v)
    }
    return out
  }
  return value
}

function mkStep(over = {}) {
  return { uid: 'u', agent: 'a', config: {}, validation: [], ...over }
}

function mkPipeline(steps) {
  return { pipeline_id: 'p', domain_profile: '', input_modality: '', description: '', parameters: {}, steps }
}

const TEXTBOOK_V1 = `pipeline_id: textbook_v1
domain_profile: education
input_modality: document
description: "Education-focused pipeline for textbook ingestion with LO extraction and graph indexing"
steps:
  - agent: text
  - agent: chunking
    config:
      chunk_size: 1024
      overlap: 100
  - agent: learning_outcome
  - agent: enrichment
  - agent: embedding
    config:
      model: "openai:text-embedding-3-large"
  - agent: kg_builder
`

test('round-trips a real profile', () => {
  const pipeline = fromYaml(TEXTBOOK_V1)
  const out = toYaml(pipeline)
  const original = stripUid(yaml.load(TEXTBOOK_V1))
  const roundTripped = stripUid(yaml.load(out))
  assert.deepStrictEqual(roundTripped, original)
})

test('keeps a stage on its steps', () => {
  const text = `steps:\n  - agent: enrichment\n    stage: enrich\n  - agent: quality\n    stage: enrich\n`
  const pipeline = fromYaml(text)
  const out = toYaml(pipeline)
  const doc = yaml.load(out)
  assert.equal(doc.steps[0].stage, 'enrich')
  assert.equal(doc.steps[1].stage, 'enrich')
})

test('groups contiguous stage steps', () => {
  const steps = [mkStep({ agent: 's1' }), mkStep({ agent: 's2', stage: 'x' }), mkStep({ agent: 's3', stage: 'x' }), mkStep({ agent: 's4' })]
  const groups = groupByStage(steps)
  assert.equal(groups.length, 3)
  assert.deepStrictEqual(groups.map((g) => g.steps.length), [1, 2, 1])
})

test('does not group a split stage', () => {
  const steps = [mkStep({ agent: 's1', stage: 'a' }), mkStep({ agent: 's2', stage: 'b' }), mkStep({ agent: 's3', stage: 'a' })]
  const groups = groupByStage(steps)
  assert.equal(groups.length, 3)
  assert.deepStrictEqual(groups.map((g) => g.steps.length), [1, 1, 1])
})

test('prunes a config value that equals the default', () => {
  const spec = {
    id: 'chunking',
    cls: 'Chunking',
    kind: 'primitive',
    group: 'g',
    fields: [
      { name: 'chunk_size', type: 'integer', required: false, default: 512 },
      { name: 'overlap', type: 'integer', required: false, default: 50 },
    ],
  }
  const out = pruneConfig({ chunk_size: 512, overlap: 99 }, spec)
  assert.deepStrictEqual(out, { overlap: 99 })
})

test('keeps a required field even at its default', () => {
  const spec = {
    id: 'agent',
    cls: 'Agent',
    kind: 'primitive',
    group: 'g',
    fields: [{ name: 'x', type: 'string', required: true, default: 'foo' }],
  }
  const out = pruneConfig({ x: 'foo' }, spec)
  assert.deepStrictEqual(out, { x: 'foo' })
})

test('rejects YAML with no steps list', () => {
  assert.throws(() => fromYaml('pipeline_id: x'), (err) => err.message.includes('steps'))
})

test('rejects a step with no agent', () => {
  assert.throws(() => fromYaml('steps:\n  - config: {}'), (err) => err.message.includes('Step 1'))
})

test('reports a split stage', () => {
  const steps = [mkStep({ agent: 's1', stage: 'a' }), mkStep({ agent: 's2', stage: 'b' }), mkStep({ agent: 's3', stage: 'a' })]
  const errors = validatePipeline(mkPipeline(steps), [])
  assert.ok(errors.some((e) => e.includes('contiguous')))
})

test('reports a missing required field', () => {
  const spec = {
    id: 'chunking',
    cls: 'Chunking',
    kind: 'primitive',
    group: 'g',
    fields: [{ name: 'chunk_size', type: 'integer', required: true }],
  }
  const steps = [mkStep({ agent: 'chunking', config: {} })]
  const errors = validatePipeline(mkPipeline(steps), [spec])
  assert.ok(errors.some((e) => e.includes('chunk_size')))
})

test('omits an empty top-level field', () => {
  const pipeline = mkPipeline([])
  pipeline.description = ''
  pipeline.parameters = {}
  const out = toYaml(pipeline)
  const doc = yaml.load(out)
  assert.ok(!('description' in doc))
  assert.ok(!('parameters' in doc))
})

test('reports two item-rebuilding steps sharing a stage', () => {
  const specs = [
    { id: 'chunking', fields: [], replaces_items: true },
    { id: 'page_chunking', fields: [], replaces_items: true },
    { id: 'extract', fields: [], replaces_items: false },
  ]
  const clash = mkPipeline([
    mkStep({ uid: 'a', agent: 'chunking', stage: 'fanout' }),
    mkStep({ uid: 'b', agent: 'page_chunking', stage: 'fanout' }),
  ])
  const errs = validatePipeline(clash, specs)
  assert.equal(errs.length, 1)
  assert.match(errs[0], /rebuilds the item list/)
  assert.match(errs[0], /chunking and page_chunking/)

  // one rebuilder beside a metadata writer is the supported pattern
  const fine = mkPipeline([
    mkStep({ uid: 'a', agent: 'chunking', stage: 'fanout' }),
    mkStep({ uid: 'b', agent: 'extract', stage: 'fanout' }),
  ])
  assert.deepEqual(validatePipeline(fine, specs), [])

  // same two steps are fine once they are in different stages
  const split = mkPipeline([
    mkStep({ uid: 'a', agent: 'chunking', stage: 'one' }),
    mkStep({ uid: 'b', agent: 'page_chunking', stage: 'two' }),
  ])
  assert.deepEqual(validatePipeline(split, specs), [])
})
