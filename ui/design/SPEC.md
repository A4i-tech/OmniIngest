# OmniIngest Pipeline Studio — UI spec

Visual rules derived from hermes-agent.nousresearch.com. Every value here is literal.
`src/theme.css` already defines the CSS variables and the shared classes. Use them.
Do not invent colors. Do not add a CSS framework.

## Absolute rules

- **No border radius anywhere.** Every edge is square.
- **No box-shadow. No gradient.** Depth comes from the darker canvas well (`--canvas`) and 1px hairlines.
- Inline `style={{...}}` is fine. A component may also add a small `<style>` block for hover states.
- Icons are inline SVG, stroke-based, 1px–1.2px stroke, on an 8/9/11px box. Never emoji.

## Colors (CSS vars, already in theme.css)

Dark-ink scheme. Three real elevation steps replace the old flat blue field.

| var | value | use |
|---|---|---|
| `--canvas` | `#080B16` | graph well, input fills (the lowest layer) |
| `--ground` | `#0B0F1E` | page and rails |
| `--blue` | `#121829` | raised surfaces: nodes, stat cells, ghost buttons |
| `--cream` | `#ECEFF7` | primary text |
| `--accent` | `#3D5AFE` | selection, focus, links, primary button |
| `--running` | `#FFB020` | running state, progress fill, live dot |
| `--jq` | `#FFD98E` | jq expressions, YAML scalars, component-node accent |
| `--failed` | `#FF5C4D` | failure, retry, destructive |

The `--cNN` names are kept from the first scheme so every existing reference resolves, but
they are **opaque now** — cream at low alpha over a dark ground washed the borders out, which
was the whole visibility problem.

| var | value | use |
|---|---|---|
| `--c88` | `#ECEFF7` | body text |
| `--c62` | `#A7B0C8` | secondary text, prose |
| `--c50` | `#8B95B2` | micro-labels |
| `--c40` | `#6B7391` | pending, disabled |
| `--c28` | `#39435F` | control and node borders |
| `--c22` | `#2A3450` | panel rules |
| `--c18` | `#222B44` | section rules |
| `--c10` | `#1A2135` | track fill |
| `--edge` | `#4A5675` | graph connectors |
| `--grid` | `#1E2740` | canvas dot grid |
| `--hover` | `#1A2237` | row hover, active row |

Status -> color: `pending` `--c40` - `running` `--running` - `success` `--cream` -
`failure` `--failed` - `skipped` `--c40` dashed. Selection is `--accent`, never `--cream`.

## Type

| role | family | size | notes |
|---|---|---|---|
| page title | `--serif` 400 | 46–58px | `line-height: 0.94` |
| panel heading | `--serif` 400 | 30px | `line-height: 1` |
| node title / values | `--mono` 500 | 12px | |
| table + form values | `--mono` 400 | 10–11px | |
| prose | `--sans` 400 | 12px | `line-height: 1.45`, color `--c62` |
| micro-label | `--mono` 400 | 9px | `letter-spacing: .2em`, uppercase, `--c50` — class `.label` |

Serif is for identity only: page titles, panel headings, big numerals. Never for UI chrome.

## Metrics

- App header 56px, `border-bottom: 1px solid var(--c22)`.
- Rail dividers `1px solid var(--c22)`. Section rules `1px solid var(--c18)`.
- Buttons 30px tall (`.btn`), 26px in canvas toolbars (`.btn .btn-sm`). Inputs 28px.
- Panel padding 18px. Canvas padding 24px. Node padding `11px 13px`.
- Builder rails: left 248px, right 400px. Runs rails: left 300px, right 372px.

## App shell (already written in App.tsx — do not change it)

Header: `OmniIngest` in `--serif` 19px/500, a 1px vertical divider, then three tabs in
`--mono` 10px `.16em` uppercase. Active tab is `--cream` with a 1px cream bottom border;
inactive is `--c50`.

## Errors

Every user-facing error names the fault and the next action, in plain language.
Never show a bare status code. `ApiError` in `src/api.ts` carries `.message` (the fault)
and `.hint` (what to do). Render both: message in `--cream`, hint below in `--c62`.
