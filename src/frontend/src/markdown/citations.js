// Citation markers in Keeper narration: `[R1]`, `[R1, R3]`, `[R1][R2]`.
//
// `remarkCitations` is a tiny remark (mdast) plugin: it splits text nodes around
// marker groups and inserts `citationRef` nodes, which remark-rehype turns into a
// custom `<cite-ref data-markers="1,3">` element (via `data.hName`). react-markdown
// then renders that element through a `components['cite-ref']` override, so the
// chips are ordinary React elements — no raw HTML is ever injected.
//
// Unresolved `[label]` references are plain text in CommonMark, so `[R1]` reaches
// the plugin as text. Code (`code`, `inlineCode`) keeps its literal value, and
// link text is skipped so a button is never nested inside an <a>.

import { createContext } from 'react'

export const CITE_TAG = 'cite-ref'

/**
 * What the inline chips of one Keeper message need to know.
 *  pending   — the message is still streaming (or was interrupted): neutral chips
 *  byMarker  — Map<number, Citation>
 *  active    — marker whose source entry is highlighted
 *  listId    — id of the sources panel (for aria-controls)
 *  onSelect  — (marker) => void, opens the sources and focuses that entry
 */
export const CitationContext = createContext({
  pending: true,
  byMarker: new Map(),
  active: null,
  listId: undefined,
  onSelect: null,
})

// `[R1]`, `[R1, R3]`, `[R1; R3]`, `[R1,3]` (a group opens with a Latin `R`).
const MARKER_GROUP = /\[\s*R\s*(\d{1,4})((?:\s*[,;]\s*R?\s*\d{1,4})*)\s*\]/g

const SKIP_TYPES = new Set(['code', 'inlineCode', 'link', 'linkReference', 'definition', 'html'])

/** `3`, `"3"`, `"R3"` → 3; anything else → null. */
export function toMarker(value) {
  if (typeof value === 'number') return Number.isInteger(value) && value > 0 ? value : null
  if (typeof value !== 'string') return null
  const m = /^\s*\[?\s*R?\s*(\d{1,4})\s*\]?\s*$/i.exec(value)
  return m ? Number(m[1]) : null
}

function groupMarkers(match) {
  const numbers = match[0].match(/\d{1,4}/g) ?? []
  return [...new Set(numbers.map(Number))]
}

/** Every marker number mentioned in a raw Keeper text (in order of first use). */
export function extractMarkers(text) {
  if (typeof text !== 'string' || !text.includes('[')) return []
  const seen = new Set()
  for (const match of text.matchAll(MARKER_GROUP)) {
    for (const n of groupMarkers(match)) seen.add(n)
  }
  return [...seen]
}

function splitText(value) {
  if (!value.includes('[')) return null
  const parts = []
  let last = 0
  for (const match of value.matchAll(MARKER_GROUP)) {
    const markers = groupMarkers(match)
    if (markers.length === 0) continue
    if (match.index > last) parts.push({ type: 'text', value: value.slice(last, match.index) })
    parts.push({
      type: 'citationRef',
      markers,
      children: [],
      data: { hName: CITE_TAG, hProperties: { dataMarkers: markers.join(',') } },
    })
    last = match.index + match[0].length
  }
  if (parts.length === 0) return null
  if (last < value.length) parts.push({ type: 'text', value: value.slice(last) })
  return parts
}

function transform(node) {
  if (!Array.isArray(node.children) || SKIP_TYPES.has(node.type)) return
  let changed = false
  const next = []
  for (const child of node.children) {
    if (child.type === 'text' && typeof child.value === 'string') {
      const parts = splitText(child.value)
      if (parts) {
        next.push(...parts)
        changed = true
        continue
      }
    } else {
      transform(child)
    }
    next.push(child)
  }
  if (changed) node.children = next
}

export default function remarkCitations() {
  return (tree) => {
    transform(tree)
  }
}

/** Marker numbers from a rendered `cite-ref` element's props. */
export function markersFromProps(props) {
  const raw = props?.['data-markers'] ?? props?.node?.properties?.dataMarkers ?? ''
  const list = Array.isArray(raw) ? raw : String(raw).split(/[\s,]+/)
  return list.map(toMarker).filter((n) => n != null)
}
