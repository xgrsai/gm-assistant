import { useMemo } from 'react'
import { X } from 'lucide-react'
import { CHARACTERISTIC_LABELS } from '../i18n/uk.js'

const CHARACTERISTIC_ORDER = ['STR', 'CON', 'SIZ', 'DEX', 'APP', 'INT', 'POW', 'EDU']

function Meter({ label, abbr, value, max, tone }) {
  const hasMax = Number.isFinite(max) && max > 0
  const pct = hasMax && Number.isFinite(value) ? Math.max(0, Math.min(100, (value / max) * 100)) : null
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-ash-300">
          {label} <span className="text-ash-600">{abbr}</span>
        </span>
        <span className="font-mono tabular-nums text-parchment">
          {value ?? '—'}
          {hasMax && <span className="text-ash-600">/{max}</span>}
        </span>
      </div>
      {pct != null && (
        <div
          className="mt-1 h-1.5 overflow-hidden rounded-full bg-ash-800"
          role="meter"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={max}
          aria-valuenow={value}
        >
          <div className={`h-full rounded-full transition-[width] duration-700 ${tone}`} style={{ width: `${pct}%` }} />
        </div>
      )}
    </div>
  )
}

function SheetBody({ investigator }) {
  const characteristics = investigator.characteristics ?? {}
  const charKeys = useMemo(() => {
    const known = CHARACTERISTIC_ORDER.filter((k) => k in characteristics)
    const extra = Object.keys(characteristics).filter((k) => !CHARACTERISTIC_ORDER.includes(k))
    return [...known, ...extra]
  }, [characteristics])

  const skills = useMemo(
    () =>
      Object.entries(investigator.skills ?? {})
        .filter(([name, v]) => name !== '...' && Number.isFinite(Number(v)) && Number(v) > 0)
        .sort((a, b) => Number(b[1]) - Number(a[1]) || a[0].localeCompare(b[0])),
    [investigator.skills],
  )

  return (
    <>
      <div className="border-b border-ash-800 px-4 pb-4">
        <h2 className="font-display text-2xl font-semibold leading-tight text-parchment">{investigator.name ?? 'Слідчий'}</h2>
        <p className="mt-0.5 text-sm text-ash-400">
          {investigator.occupation}
          {investigator.age != null && <> · {investigator.age} р.</>}
        </p>
      </div>

      <section className="space-y-3 border-b border-ash-800 px-4 py-4" aria-label="Стан">
        <Meter label="Здоров’я" abbr="HP" value={investigator.hp} max={investigator.hp_max} tone="bg-blood" />
        <Meter label="Магія" abbr="MP" value={investigator.mp} max={investigator.mp_max} tone="bg-verdigris/80" />
        <Meter label="Розсудливість" abbr="SAN" value={investigator.san} max={investigator.san_max} tone="bg-ember" />
        <Meter label="Удача" abbr="Luck" value={investigator.luck} max={99} tone="bg-brass/80" />
      </section>

      {charKeys.length > 0 && (
        <section className="border-b border-ash-800 px-4 py-4" aria-labelledby="sheet-chars">
          <h3 id="sheet-chars" className="mb-2 text-xs uppercase tracking-widest text-ash-500">
            Характеристики
          </h3>
          <dl className="grid grid-cols-4 gap-1.5">
            {charKeys.map((key) => {
              const v = Number(characteristics[key])
              return (
                <div
                  key={key}
                  className="rounded border border-ash-800 bg-ash-900/70 px-1 py-1.5 text-center"
                  title={CHARACTERISTIC_LABELS[key] ?? key}
                >
                  <dt className="text-[10px] uppercase tracking-wider text-ash-500">{key}</dt>
                  <dd className="font-mono text-base leading-tight tabular-nums text-parchment">{characteristics[key]}</dd>
                  {Number.isFinite(v) && (
                    <dd className="font-mono text-[10px] text-ash-600" aria-label="половина та п’ята частина">
                      {Math.floor(v / 2)}·{Math.floor(v / 5)}
                    </dd>
                  )}
                </div>
              )
            })}
          </dl>
        </section>
      )}

      {skills.length > 0 && (
        <section className="px-4 py-4" aria-labelledby="sheet-skills">
          <h3 id="sheet-skills" className="mb-2 text-xs uppercase tracking-widest text-ash-500">
            Навички
          </h3>
          <ul className="space-y-0.5 text-sm">
            {skills.map(([name, value]) => (
              <li key={name} className="flex items-baseline justify-between gap-2 border-b border-dotted border-ash-800 py-0.5">
                <span className="min-w-0 truncate text-ash-200" title={name}>
                  {name}
                </span>
                <span className="font-mono text-xs tabular-nums text-parchment">{value}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  )
}

/**
 * Right-side character sheet. Static on `xl+`; a slide-over drawer below that.
 */
export default function InvestigatorPanel({ investigator, open, onClose }) {
  return (
    <>
      <div
        className={`fixed inset-0 z-30 bg-black/60 transition-opacity xl:hidden ${open ? 'opacity-100' : 'pointer-events-none opacity-0'}`}
        onClick={onClose}
        aria-hidden
      />
      <aside
        id="investigator-panel"
        aria-label="Аркуш слідчого"
        className={`fixed inset-y-0 right-0 z-40 flex w-[min(20rem,88vw)] flex-col border-l border-ash-800 bg-ash-900 transition-[transform,visibility] duration-300
          xl:visible xl:static xl:z-auto xl:w-72 xl:translate-x-0 xl:bg-ash-900/40 xl:transition-none
          ${open ? 'visible translate-x-0' : 'invisible translate-x-full'}`}
      >
        <div className="flex shrink-0 items-center justify-between px-4 pb-2 pt-4">
          <span className="text-xs uppercase tracking-widest text-ash-500">Аркуш слідчого</span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Закрити аркуш слідчого"
            className="rounded p-1 text-ash-400 hover:bg-ash-800 hover:text-parchment xl:hidden"
          >
            <X size={16} aria-hidden />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden pb-6">
          {investigator ? (
            <SheetBody investigator={investigator} />
          ) : (
            <p className="px-4 py-6 text-sm italic text-ash-500">Аркуш з’явиться, щойно справу буде відкрито.</p>
          )}
        </div>
      </aside>
    </>
  )
}
