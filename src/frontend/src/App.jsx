import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import MessageInput from './components/MessageInput'
import { sendMessage } from './lib/api'

export default function App() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSend = async (text) => {
    const userMsg = { role: 'user', content: text }
    const updated = [...messages, userMsg]
    setMessages(updated)
    setLoading(true)
    setError(null)

    try {
      const reply = await sendMessage(updated)
      setMessages([...updated, { role: 'assistant', content: reply }])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleNewGame = () => {
    setMessages([])
    setError(null)
  }

  return (
    <div className="h-full flex bg-ink">
      <Sidebar onNewGame={handleNewGame} />

      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="px-6 py-4 border-b border-ash-800 flex items-center justify-between shrink-0">
          <div>
            <h1 className="font-display text-sm tracking-widest text-parchment uppercase">
              The Ashen Crown
            </h1>
            <p className="text-xs text-ash-500 mt-0.5">Dark fantasy campaign</p>
          </div>
          {error && (
            <p className="text-sm text-red-400 bg-red-900/20 border border-red-800/40 px-3 py-1.5 rounded-lg">
              {error}
            </p>
          )}
        </header>

        <ChatWindow messages={messages} loading={loading} />
        <MessageInput onSend={handleSend} disabled={loading} />
      </main>
    </div>
  )
}
