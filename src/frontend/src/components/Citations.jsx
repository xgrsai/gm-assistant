import { useContext, useEffect, useRef } from 'react'
import { AlertTriangle, BookOpen, ChevronDown } from 'lucide-react'
import { CitationContext, markersFromProps } from '../markdown/citations.js'
import { CITATION_INVALID_HINT, SOURCE_TYPE_LABELS } from '../i18n/uk.js'

function sectionPath(section) {
  return typeof section === 'string' ? section.split(/\s*>\s*/).filter(Boolean).join(' › ') : ''
}

function scorePercent(score) {
  const n = Number(score)
  return Number.isFinite(n) ? `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%` : null
}

function CitationChip({ marker }) {
  const { pending, byMarker, active, listId, onSelect } = useContext(CitationContext)
  const label = `R${marker}`

  if (pending || !onSelect) {
    return <span className="cite-chip cite-chip--pending">{label}</span>
  }

  const citation = byMarker.get(marker)
  if (!citation) {
    return (
      <button
        type="button"
        onClick={() => onSelect(marker)}
        className="cite-chip cite-chip--invalid"
        title={CITATION_INVALID_HINT}
        aria-label={`Джерело ${label}: ${CITATION_INVALID_HINT}`}
        aria-controls={listId}
      >
        {label}
      </button>
    )
  }

  const where = [citation.source, sectionPath(citation.section)].filter(Boolean).join(' › ')
  return (
    <button
      type="button"
      onClick={() => onSelect(marker)}
      className={`cite-chip cite-chip--valid ${active === marker ? 'cite-chip--active' : ''}`}
      title={where || undefined}
      aria-label={`Джерело ${label}`}
      aria-controls={listId}
    >
      {label}
    </button>
  )
}

/** react-markdown override for the `<cite-ref>` element produced by `remarkCitations`. */
export function CitationRef(props) {
  const markers = markersFromProps(props)
  if (markers.length === 0) return null
  return (
    <span className="cite-ref">
      {markers.map((m) => (
        <CitationChip key={m} marker={m} />
      ))}
    </span>
  )
}

/**
 * Collapsible «Джерела (N)» block under a Keeper bubble.
 *
 * @param {object} props
 * @param {Array} props.citations        sorted by marker
 * @param {number[]} props.invalid       markers cited in the text but not retrieved
 * @param {boolean} props.open
 * @param {(open: boolean) => void} props.onToggle
 * @param {{marker: number, seq: number}|null} props.focus  entry to scroll to / highlight
 * @param {string} props.listId
 */
export function SourcesSection({ citations, invalid, open, onToggle, focus, listId }) {
  const entryRefs = useRef(new Map())
  const warningRef = useRef(null)
  const panelRef = useRef(null)
  const count = citations.length

  // Bring the requested entry (or the warning for an unknown marker) into view.
  useEffect(() => {
    if (!open) return
    const target = focus ? (entryRefs.current.get(focus.marker) ?? warningRef.current) : panelRef.current
    if (!target) return
    target.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    if (focus) target.focus({ preventScroll: true })
  }, [open, focus])

  return (
    <div className="mt-1.5 w-full">
      <button
        type="button"
        onClick={() => onToggle(!open)}
        aria-expanded={open}
        aria-controls={listId}
        className="inline-flex items-center gap-1.5 rounded px-1.5 py-1 text-xs text-ash-400 transition-colors hover:bg-ash-800/60 hover:text-parchment"
      >
        <BookOpen size={13} className="text-brass/80" aria-hidden />
        <span className="uppercase tracking-widest">Джерела</span>
        <span className="font-mono tabular-nums text-ash-500">({count})</span>
        {invalid.length > 0 && (
          <span className="inline-flex items-center gap-0.5 text-red-300/80">
            <AlertTriangle size={11} aria-hidden />
            <span className="sr-only">, посилань без джерела:</span>
            <span className="font-mono tabular-nums">{invalid.length}</span>
          </span>
        )}
        <ChevronDown size={13} className={`transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>

      {open && (
        <div
          id={listId}
          ref={panelRef}
          className="mt-1 animate-fade-in overflow-hidden rounded-lg border border-ash-700/60 bg-ash-900/70"
        >
          {invalid.length > 0 && (
            <p
              ref={warningRef}
              tabIndex={-1}
              className={`flex items-start gap-2 border-b border-ash-800 px-3 py-2 text-xs leading-relaxed text-red-300/90 focus:outline-none
                ${focus && invalid.includes(focus.marker) ? 'bg-blood/15' : ''}`}
            >
              <AlertTriangle size={13} className="mt-0.5 shrink-0" aria-hidden />
              <span>
                {invalid.map((m) => `R${m}`).join(', ')}: {CITATION_INVALID_HINT}.
              </span>
            </p>
          )}

          {count === 0 ? (
            <p className="px-3 py-2 text-xs italic text-ash-500">Жодного фрагмента бази знань не процитовано.</p>
          ) : (
            <ol className="divide-y divide-ash-800" aria-label="Процитовані фрагменти бази знань">
              {citations.map((c) => {
                const active = focus?.marker === c.marker
                const path = sectionPath(c.section)
                const pct = scorePercent(c.score)
                return (
                  <li
                    key={c.marker}
                    ref={(el) => {
                      if (el) entryRefs.current.set(c.marker, el)
                      else entryRefs.current.delete(c.marker)
                    }}
                    tabIndex={-1}
                    className={`flex items-start gap-2.5 border-l-2 px-3 py-2.5 transition-colors focus:outline-none
                      ${active ? 'border-brass bg-brass/10' : 'border-transparent'}`}
                  >
                    <span className="cite-chip cite-chip--valid mt-0.5 shrink-0 !cursor-default">
                      R{c.marker}
                      <span className="sr-only">:</span>
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm leading-snug text-parchment">
                        {c.source || 'Невідоме джерело'}
                        {path && <span className="text-ash-400"> › {path}</span>}
                      </p>
                      <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[11px] text-ash-500">
                        {c.source_type && (
                          <span className="uppercase tracking-wider">{SOURCE_TYPE_LABELS[c.source_type] ?? c.source_type}</span>
                        )}
                        {pct && (
                          <span title="Косинусна схожість фрагмента із запитом" className="font-mono tabular-nums text-ash-600">
                            схожість {pct}
                          </span>
                        )}
                      </p>
                      {c.excerpt && (
                        <p className="mt-1.5 whitespace-pre-line border-l border-ash-700 pl-2.5 text-xs italic leading-relaxed text-ash-300">
                          {c.excerpt}
                        </p>
                      )}
                    </div>
                  </li>
                )
              })}
            </ol>
          )}
        </div>
      )}
    </div>
  )
}
