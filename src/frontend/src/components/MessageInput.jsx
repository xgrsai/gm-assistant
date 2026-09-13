import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { SendHorizonal } from 'lucide-react'

const MAX_HEIGHT_PX = 176
const MAX_LENGTH = 4000

function autoSize(el) {
  if (!el) return
  el.style.height = 'auto'
  const full = el.scrollHeight
  el.style.height = `${Math.min(full, MAX_HEIGHT_PX)}px`
  el.style.overflowY = full > MAX_HEIGHT_PX ? 'auto' : 'hidden'
}

/**
 * Typing stays enabled while the Keeper streams; only sending is blocked.
 * `onSend(text)` resolves `false` when the request was refused before the
 * stream started — the text is then put back.
 */
export default function MessageInput({ onSend, canSend, hint }) {
  const [value, setValue] = useState('')
  const [sending, setSending] = useState(false)
  const textareaRef = useRef(null)

  // JS auto-grow (field-sizing: content is not supported everywhere).
  useLayoutEffect(() => {
    autoSize(textareaRef.current)
  }, [value])

  // Re-measure when the available width changes (layout settling, drawers, rotation).
  useEffect(() => {
    const el = textareaRef.current
    if (!el || typeof ResizeObserver !== 'function') return
    let lastWidth = el.clientWidth
    const ro = new ResizeObserver(() => {
      if (el.clientWidth === lastWidth) return
      lastWidth = el.clientWidth
      autoSize(el)
    })
    ro.observe(el)
    document.fonts?.ready?.then(() => autoSize(el))
    return () => ro.disconnect()
  }, [])

  const trimmed = value.trim()
  const sendDisabled = !canSend || sending || !trimmed

  const submit = async () => {
    if (sendDisabled) return
    const text = value
    setValue('')
    setSending(true)
    let accepted = false
    try {
      accepted = await onSend(text.trim())
    } finally {
      setSending(false)
    }
    if (!accepted) {
      // Keep what the player wrote; don't overwrite anything typed meanwhile.
      setValue((current) => (current ? current : text))
    }
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key !== 'Enter' || e.shiftKey) return
    if (e.nativeEvent.isComposing || e.keyCode === 229) return // IME composition
    e.preventDefault()
    submit()
  }

  return (
    <div className="shrink-0 border-t border-ash-800 bg-ash-900/40 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 sm:px-6">
      <form
        className="flex items-end gap-2 rounded-xl border border-ash-700 bg-ash-800/60 px-3 py-2 transition-colors focus-within:border-ash-500 sm:gap-3 sm:px-4"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Опишіть дію свого слідчого…"
          aria-label="Дія слідчого"
          rows={1}
          maxLength={MAX_LENGTH}
          className="min-w-0 flex-1 resize-none bg-transparent py-1 font-body text-base leading-relaxed text-parchment placeholder:text-ash-600 focus:outline-none"
        />
        <button
          type="submit"
          disabled={sendDisabled}
          aria-label="Надіслати дію"
          title="Надіслати (Enter)"
          className="mb-0.5 shrink-0 rounded-lg p-2 text-ember transition-colors hover:bg-ember/20 hover:text-parchment disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-ember"
        >
          <SendHorizonal size={18} aria-hidden />
        </button>
      </form>
      <p className="mt-1.5 hidden text-center text-xs text-ash-600 sm:block" aria-live="polite">
        {hint || 'Enter — надіслати · Shift+Enter — новий рядок'}
      </p>
    </div>
  )
}
