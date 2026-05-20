import { useState, useRef } from 'react'
import { SendHorizonal } from 'lucide-react'

export default function MessageInput({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  const handleSend = () => {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue('')
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="px-6 py-4 border-t border-ash-800 bg-ash-900/40">
      <div className="flex items-end gap-3 bg-ash-800/60 border border-ash-700 rounded-xl px-4 py-3 focus-within:border-ash-500 transition-colors">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder="Describe your action…"
          rows={1}
          className="flex-1 bg-transparent resize-none text-base text-parchment placeholder:text-ash-600
            focus:outline-none font-body leading-relaxed max-h-36 overflow-y-auto disabled:opacity-50"
          style={{ fieldSizing: 'content' }}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="shrink-0 p-1.5 rounded-lg text-ember hover:text-parchment hover:bg-ember/20
            disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <SendHorizonal size={18} />
        </button>
      </div>
      <p className="text-xs text-ash-600 mt-2 text-center">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  )
}
