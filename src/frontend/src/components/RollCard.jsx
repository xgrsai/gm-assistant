import { Brain, Dices } from 'lucide-react'
import { landingMs, useDiceRoll } from '../hooks/useDiceRoll.js'
import { DIFFICULTY_LABELS, LEVEL_LABELS, ROLL_KIND_LABELS } from '../i18n/uk.js'

const LEVEL_STYLES = {
  critical: 'text-brass border-brass/60 bg-brass/10',
  extreme: 'text-verdigris border-verdigris/60 bg-verdigris/15',
  hard: 'text-verdigris border-verdigris/50 bg-verdigris/10',
  regular: 'text-verdigris border-verdigris/40 bg-verdigris/5',
  failure: 'text-ash-300 border-ash-600 bg-ash-800/60',
  fumble: 'text-red-300 border-blood/80 bg-blood/25',
}

// One-off accent when a live roll lands on an extreme outcome.
const LANDING_ACCENT = {
  critical: 'animate-critical-glow',
  fumble: 'animate-fumble-shake',
}

// Temporary insanity is decided by the INT roll that follows the SAN roll, so it is shown on that card.
const INSANITY_FLAGS = [
  ['involuntary_action', 'Мимовільна дія'],
  ['indefinite_insanity', 'Безстрокове божевілля'],
  ['permanent_insanity', 'Постійне божевілля'],
]

const PURPOSE_LABELS = {
  temporary_insanity_check: 'перевірка тимчасового божевілля',
}

/** Tens die of the final result: 100 is "00" + "0". */
function chosenTens(roll) {
  if (typeof roll !== 'number') return null
  return roll === 100 ? 0 : Math.floor(roll / 10)
}

function Die({ children, active = true, title, tumbling = false, landed = false }) {
  const motion = tumbling ? 'animate-die-tumble' : landed ? 'animate-die-land' : ''
  return (
    <span
      title={title}
      className={`inline-flex h-6 min-w-[1.75rem] items-center justify-center rounded border px-1 font-mono text-xs tabular-nums ${motion}
        ${active ? 'border-ash-500 bg-ash-800 text-parchment' : 'border-ash-700 bg-transparent text-ash-500 line-through decoration-ash-600'}`}
    >
      {children}
    </span>
  )
}

/**
 * @param {object} props
 * @param {object} props.roll       roll record from the server
 * @param {boolean} [props.animate] fade the card in
 * @param {boolean} [props.live]    the roll just arrived on the stream: tumble the dice before showing the result
 * @param {number} [props.delayMs]  wait before tumbling, so several rolls of one turn land one after another
 * @param {() => void} [props.onSettled] called once the result is shown
 */
export default function RollCard({ roll, animate = true, live = false, delayMs = 0, onSettled }) {
  const { enabled, phase, elapsed, faces } = useDiceRoll(roll, { live, delayMs, onSettled })
  if (!roll || typeof roll !== 'object') return null
  const waiting = phase === 'waiting'
  const hidden = phase !== 'settled' // the result is not revealed yet
  const isSanity = roll.kind === 'sanity'
  const isInsanityCheck = roll.purpose === 'temporary_insanity_check'
  const level = roll.level in LEVEL_STYLES ? roll.level : roll.passed ? 'regular' : 'failure'
  const levelLabel = LEVEL_LABELS[level] ?? roll.level
  const difficulty = DIFFICULTY_LABELS[roll.difficulty] ?? roll.difficulty
  const tens = Array.isArray(roll.tens) ? roll.tens : []
  const pick = chosenTens(roll.roll)
  const pickIndex = tens.indexOf(pick)
  const bonus = Number(roll.bonus) || 0
  const penalty = Number(roll.penalty) || 0
  const value = Number(roll.value)
  const flags = INSANITY_FLAGS.filter(([key]) => roll[key])
  const Icon = isSanity ? Brain : Dices
  const kindLabel = ROLL_KIND_LABELS[roll.kind] ?? roll.kind

  // Tens dice land first, the units die last; until a die lands it shows a random face.
  const tensLanded = (i) => !hidden || elapsed >= landingMs(i)
  const unitsLanded = !hidden || elapsed >= landingMs(tens.length)
  const tensFace = (t, i) => (tensLanded(i) ? t : (faces?.tens[i] ?? 0))
  const unitsFace = unitsLanded ? roll.units : (faces?.units ?? 0)
  const allLanded = unitsLanded && tens.every((_, i) => tensLanded(i))
  let total = roll.roll
  if (!allLanded) {
    // "00" + "0" reads as 100, like the real dice
    total = tens.length && roll.units != null ? tensFace(tens[0], 0) * 10 + unitsFace || 100 : '··'
  }
  const landing = !hidden && enabled

  return (
    <figure
      className={`w-full max-w-md ${waiting ? 'invisible' : animate ? 'animate-fade-in' : ''} rounded-lg border border-ash-700/70 bg-ash-900/80 px-3 py-2.5 text-sm shadow-[inset_0_1px_0_rgba(232,220,200,0.03)]`}
      aria-label={`Кидок: ${roll.target ?? kindLabel}, результат ${roll.roll}, ${levelLabel}`}
      aria-busy={hidden || undefined}
    >
      <div className={landing ? (LANDING_ACCENT[level] ?? '') : ''}>
        <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1.5">
          <div className="flex min-w-0 items-center gap-2">
            <Icon
              size={15}
              className={`shrink-0 ${isSanity ? 'text-ember' : 'text-ash-400'} ${hidden && !waiting ? 'animate-die-tumble' : ''}`}
              aria-hidden
            />
            <div className="min-w-0">
              <p className="truncate font-display text-base font-semibold leading-tight text-parchment">
                {roll.target ?? kindLabel}
                {Number.isFinite(value) && <span className="ml-1.5 font-body text-sm font-normal text-ash-400">{value}</span>}
              </p>
              <p className="text-xs text-ash-500">
                {kindLabel}
                {difficulty && <> · {difficulty} складність</>}
                {roll.purpose && <> · {PURPOSE_LABELS[roll.purpose] ?? roll.purpose}</>}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            {roll.pushed && (
              <span className="rounded border border-ember/50 bg-ember/10 px-1.5 py-0.5 text-[11px] uppercase tracking-wider text-ember">
                Форсований
              </span>
            )}
            {hidden ? (
              <span className="rounded border border-ash-700 px-2 py-0.5 text-xs italic text-ash-500">кидок…</span>
            ) : (
              <span
                className={`rounded border px-2 py-0.5 text-xs font-semibold ${LEVEL_STYLES[level]} ${landing ? 'animate-badge-pop' : ''}`}
              >
                {levelLabel}
              </span>
            )}
          </div>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-[11px] text-ash-500">d100</span>
            <span
              className={`inline-block min-w-[2.5ch] font-mono text-2xl font-semibold tabular-nums leading-none
                ${hidden ? 'text-ash-400' : 'text-parchment'} ${landing ? 'animate-die-land' : ''}`}
            >
              {total}
            </span>
            {Number.isFinite(value) && <span className="font-mono text-xs text-ash-500">/ {value}</span>}
          </div>
          {(tens.length > 0 || roll.units != null) && (
            <div className="flex items-center gap-1" aria-label="Кубики">
              {tens.map((t, i) => (
                <Die
                  key={i}
                  active={hidden || pickIndex === -1 || i === pickIndex}
                  title="Десятки"
                  tumbling={!tensLanded(i)}
                  landed={enabled && tensLanded(i)}
                >
                  {`${tensFace(t, i)}0`}
                </Die>
              ))}
              {roll.units != null && (
                <>
                  <span className="px-0.5 text-ash-600">+</span>
                  <Die title="Одиниці" tumbling={!unitsLanded} landed={enabled && unitsLanded}>
                    {unitsFace}
                  </Die>
                </>
              )}
            </div>
          )}
          {(bonus > 0 || penalty > 0) && (
            <div className="flex gap-1.5 text-xs">
              {bonus > 0 && <span className="text-verdigris">+{bonus} бонусн. кубик</span>}
              {penalty > 0 && <span className="text-red-300">−{penalty} штрафн. кубик</span>}
            </div>
          )}
        </div>

        {Number.isFinite(value) && !isSanity && (
          <p className="mt-1.5 flex flex-wrap gap-x-3 text-[11px] text-ash-600">
            {[
              ['regular', value],
              ['hard', Math.floor(value / 2)],
              ['extreme', Math.floor(value / 5)],
            ].map(([key, threshold]) => (
              <span key={key} className={roll.difficulty === key ? 'text-ash-300' : undefined}>
                {DIFFICULTY_LABELS[key]} <span className="font-mono">≤{threshold}</span>
              </span>
            ))}
          </p>
        )}

        {!hidden && isInsanityCheck && (
          <p className={`mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-ash-800 pt-2 text-xs ${landing ? 'animate-fade-in' : ''}`}>
            {roll.passed ? (
              <>
                <span className="rounded border border-blood/70 bg-blood/20 px-1.5 py-0.5 text-[11px] text-red-200">
                  Тимчасове божевілля
                </span>
                <span className="text-ash-500">успіх: розум збагнув побачене</span>
              </>
            ) : (
              <span className="text-ash-400">Розум відкидає побачене — божевілля немає</span>
            )}
          </p>
        )}

        {!hidden && isSanity && (roll.san_loss != null || roll.san_after != null || flags.length > 0) && (
          <div className={`mt-2 border-t border-ash-800 pt-2 ${landing ? 'animate-fade-in' : ''}`}>
            <p className="text-xs text-ash-300">
              {roll.san_loss != null && (
                <>
                  Втрата Розсудливості: <span className="font-mono font-semibold text-ember">−{roll.san_loss}</span>
                </>
              )}
              {roll.san_after != null && (
                <>
                  {' '}→ <span className="font-mono font-semibold text-parchment">{roll.san_after}</span>
                </>
              )}
            </p>
            {flags.length > 0 && (
              <ul className="mt-1.5 flex flex-wrap gap-1.5">
                {flags.map(([key, label]) => (
                  <li key={key} className="rounded border border-blood/70 bg-blood/20 px-1.5 py-0.5 text-[11px] text-red-200">
                    {label}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </figure>
  )
}
