import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'

const WELCOME = `*The fire crackles. Shadows shift on the stone walls.*

You stand at the **threshold** — where your story begins. The world is vast, dangerous, and alive.

What do you do?`

export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
      {/* Welcome message when empty */}
      {messages.length === 0 && (
        <MessageBubble
          message={{ role: 'assistant', content: WELCOME }}
        />
      )}

      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}

      {loading && <TypingIndicator />}

      <div ref={bottomRef} />
    </div>
  )
}
