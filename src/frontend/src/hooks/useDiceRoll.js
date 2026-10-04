import { useEffect, useMemo, useRef, useState } from 'react'

/*
 * Visual-only tumble for a roll the server has already resolved: faces show random digits,
 * then each die lands on its real value in turn. Nothing here decides an outcome.
 */

const FIRST_LANDING_MS = 600 // the first die lands after this…
const LANDING_STAGGER_MS = 160 // …each further die this much later (tens dice first, units last)
const SETTLE_PAUSE_MS = 140 // beat between the last die and the verdict
const MIN_FRAME_MS = 45 // faces change this often at the start…
const MAX_FRAME_MS = 140 // …slowing down to this just before landing

export function diceCount(roll) {
  return (Array.isArray(roll?.tens) && roll.tens.length ? roll.tens.length : 1) + 1
}

/** When die `i` lands, in ms from the start of the tumble. */
export function landingMs(i) {
  return FIRST_LANDING_MS + i * LANDING_STAGGER_MS
}

export function rollDurationMs(roll) {
  return landingMs(diceCount(roll) - 1) + SETTLE_PAUSE_MS
}

function motionAllowed() {
  try {
    return !window.matchMedia('(prefers-reduced-motion: reduce)').matches && !document.hidden
  } catch {
    return false
  }
}

const randomDigit = () => Math.floor(Math.random() * 10)

/**
 * @param {object} roll  roll record from the server (`tens`, `units`, `roll`)
 * @param {{live?: boolean, delayMs?: number, onSettled?: () => void}} options
 *   live: animate (only rolls that just arrived on the stream; history never re-rolls)
 *   delayMs: wait before tumbling, so several rolls of one turn land one after another
 * @returns {{enabled: boolean, phase: 'waiting'|'rolling'|'settled', elapsed: number,
 *            faces: {tens: number[], units: number} | null}}
 */
export function useDiceRoll(roll, { live = false, delayMs = 0, onSettled } = {}) {
  const [enabled] = useState(() => live && motionAllowed())
  const [phase, setPhase] = useState(enabled ? (delayMs > 0 ? 'waiting' : 'rolling') : 'settled')
  const [elapsed, setElapsed] = useState(enabled ? 0 : Infinity)
  const settledRef = useRef(onSettled)
  settledRef.current = onSettled
  const duration = rollDurationMs(roll)
  const tensCount = Array.isArray(roll?.tens) ? roll.tens.length : 0

  useEffect(() => {
    if (!enabled) {
      settledRef.current?.()
      return undefined
    }
    let raf = 0
    let start = 0
    let lastFrame = 0
    let done = false
    const settle = () => {
      if (done) return
      done = true
      cancelAnimationFrame(raf)
      setElapsed(Infinity)
      setPhase('settled')
      settledRef.current?.()
    }
    // Wall-clock timers decide when the tumble starts and ends, so several cards stay in sequence and
    // nothing hangs if animation frames are throttled; frames only refresh the random faces.
    const frame = (now) => {
      const t = now - start
      if (t >= duration) return
      if (now - lastFrame >= MIN_FRAME_MS + (MAX_FRAME_MS - MIN_FRAME_MS) * (t / duration)) {
        lastFrame = now
        setElapsed(t)
      }
      raf = requestAnimationFrame(frame)
    }
    const begin = setTimeout(() => {
      start = performance.now()
      setPhase('rolling')
      raf = requestAnimationFrame(frame)
    }, delayMs)
    const end = setTimeout(settle, delayMs + duration)
    return () => {
      clearTimeout(begin)
      clearTimeout(end)
      cancelAnimationFrame(raf)
    }
    // Once per mounted card: re-renders (new tokens, new rolls) must not restart the tumble.
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // New random faces only when the tumble advances, not on every parent re-render.
  const faces = useMemo(
    () => (phase === 'settled' ? null : { tens: Array.from({ length: tensCount }, randomDigit), units: randomDigit() }),
    [phase, elapsed, tensCount],
  )

  return { enabled, phase, elapsed, faces }
}
