const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function sendMessage(messages) {
  const res = await fetch(`${API_URL}/api/chat/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || `HTTP ${res.status}`)
  }

  const data = await res.json()
  return data.content
}
