import { memo, useCallback, useId, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Eye, Search } from 'lucide-react'
import remarkCitations, { CITE_TAG, CitationContext, extractMarkers, toMarker } from '../markdown/citations.js'
import { CitationRef, SourcesSection } from './Citations'

const MARKDOWN_COMPONENTS = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer nofollow" />,
  [CITE_TAG]: CitationRef,
}

const DISALLOWED = ['img']
const REMARK_PLUGINS = [remarkCitations]

export const KeeperMarkdown = memo(function KeeperMarkdown({ text }) {
  return (
    <ReactMarkdown
      disallowedElements={DISALLOWED}
      unwrapDisallowed
      remarkPlugins={REMARK_PLUGINS}
      components={MARKDOWN_COMPONENTS}
    >
      {text}
    </ReactMarkdown>
  )
})

export function KeeperAvatar() {
  return (
    <div
      className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-ember/40 bg-ember/15"
      aria-hidden
    >
      <Eye size={14} className="text-ember" />
    </div>
  )
}

const EMPTY = []

/** Normalised `citations` (by marker) and the set of markers without a retrieved chunk. */
function useCitations(message, text) {
  const citations = Array.isArray(message?.citations) ? message.citations : EMPTY
  const invalidMarkers = Array.isArray(message?.invalid_markers) ? message.invalid_markers : EMPTY

  return useMemo(() => {
    const byMarker = new Map()
    for (const c of citations) {
      const marker = toMarker(c?.marker)
      if (marker != null && !byMarker.has(marker)) byMarker.set(marker, { ...c, marker })
    }
    const invalid = new Set(invalidMarkers.map(toMarker).filter((n) => n != null && !byMarker.has(n)))
    for (const n of extractMarkers(text)) if (!byMarker.has(n)) invalid.add(n)
    return {
      byMarker,
      sorted: [...byMarker.values()].sort((a, b) => a.marker - b.marker),
      invalid: [...invalid].sort((a, b) => a - b),
    }
  }, [citations, invalidMarkers, text])
}

function KeeperBubble({ message, text, streaming, interrupted, fade }) {
  const listId = `sources-${useId().replace(/:/g, '')}`
  const pending = streaming || interrupted
  const { byMarker, sorted, invalid } = useCitations(pending ? null : message, text)
  const hasSources = !pending && (sorted.length > 0 || invalid.length > 0)

  const [open, setOpen] = useState(false)
  const [focus, setFocus] = useState(null) // {marker, seq}
  const seq = useRef(0)

  const onSelect = useCallback((marker) => {
    setOpen(true)
    setFocus({ marker, seq: ++seq.current })
  }, [])

  const onToggle = useCallback((next) => {
    setOpen(next)
    if (!next) setFocus(null)
  }, [])

  const ctx = useMemo(
    () => ({ pending, byMarker, active: open ? (focus?.marker ?? null) : null, listId, onSelect: hasSources ? onSelect : null }),
    [pending, byMarker, open, focus, listId, hasSources, onSelect],
  )

  return (
    <div className={`flex justify-start ${fade}`}>
      <div className="flex min-w-0 max-w-full items-start gap-2 sm:max-w-[88%]">
        <KeeperAvatar />
        <div className="sr-only">Хранитель:</div>
        <div className="flex min-w-0 flex-col items-start">
          <div
            className={`prose-keeper min-w-0 max-w-full rounded-xl rounded-tl-sm border px-4 py-3
              ${interrupted ? 'border-ash-700/40 bg-ash-900/50 text-ash-400' : 'border-ash-700/60 bg-ash-900'}
              ${streaming ? 'stream-caret' : ''}`}
            aria-busy={streaming || undefined}
          >
            <CitationContext.Provider value={ctx}>
              <KeeperMarkdown text={text} />
            </CitationContext.Provider>
            {interrupted && <p className="!mt-2 text-xs italic text-ash-500">(розповідь перервано)</p>}
          </div>
          {hasSources && (
            <SourcesSection
              citations={sorted}
              invalid={invalid}
              open={open}
              onToggle={onToggle}
              focus={focus}
              listId={listId}
            />
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * @param {{message: {role: string, text: string, citations?: Array, invalid_markers?: Array},
 *          streaming?: boolean, interrupted?: boolean, animate?: boolean}} props
 */
function MessageBubble({ message, streaming = false, interrupted = false, animate = true }) {
  const fade = animate ? 'animate-fade-in' : ''
  const text = typeof message?.text === 'string' ? message.text : ''

  if (message?.role === 'player') {
    return (
      <div className={`flex justify-end ${fade}`}>
        <div className="flex max-w-[92%] items-start gap-2 sm:max-w-[78%]">
          <div className="sr-only">Слідчий:</div>
          <div
            className={`min-w-0 whitespace-pre-wrap break-words rounded-xl rounded-tr-sm border border-ash-700 bg-ash-800 px-4 py-3 leading-relaxed text-parchment
              ${message.pending ? 'opacity-80' : ''}`}
          >
            {text}
          </div>
          <div
            className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ash-700"
            aria-hidden
          >
            <Search size={13} className="text-ash-300" />
          </div>
        </div>
      </div>
    )
  }

  return <KeeperBubble message={message} text={text} streaming={streaming} interrupted={interrupted} fade={fade} />
}

export default memo(MessageBubble)
