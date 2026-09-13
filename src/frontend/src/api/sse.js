// POST Server-Sent Events over fetch + ReadableStream.
// `EventSource` cannot POST, so the turn endpoints are read manually.
// Frames: `event: <name>\ndata: <json>\n\n`; lines starting with `:` are keep-alive pings.

import { ApiError, errorFromResponse, networkError } from './http.js'

/**
 * Sends the request and resolves once the server has accepted the stream.
 * Throws an ApiError (with `status` and `detail`) on a pre-stream non-2xx
 * response or a network failure; rethrows AbortError untouched.
 */
export async function openEventStream(url, { body, signal, method = 'POST' } = {}) {
  let res
  try {
    res = await fetch(url, {
      method,
      signal,
      cache: 'no-store',
      headers: {
        Accept: 'text/event-stream',
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (err) {
    throw networkError(err)
  }
  if (!res.ok) throw await errorFromResponse(res)
  if (!res.body) throw new ApiError('Response has no body', { status: res.status })
  return res
}

/** Thrown when no bytes (not even a keep-alive ping) arrive for `idleTimeoutMs`. */
export class StreamIdleTimeoutError extends Error {
  constructor(ms) {
    super(`No data received for ${ms} ms`)
    this.name = 'StreamIdleTimeoutError'
  }
}

/**
 * Async generator yielding `{event, data, raw}` for every dispatched frame of
 * an already-opened response. `data` is parsed JSON (or the raw string when it
 * is not JSON). Stops when the body ends; throws AbortError when `signal` aborts
 * and StreamIdleTimeoutError when the connection is silent for `idleTimeoutMs`.
 */
export async function* readEventStream(response, { signal, idleTimeoutMs = 0 } = {}) {
  let buffer = ''
  for await (const chunk of decodeText(response.body, idleTimeoutMs)) {
    if (signal?.aborted) throw abortError(signal)
    buffer += chunk

    // Normalise CRLF / CR to LF. A trailing lone `\r` may be the first half of
    // a `\r\n` split across chunks, so hold it back until the next chunk.
    const heldCR = buffer.endsWith('\r')
    let text = (heldCR ? buffer.slice(0, -1) : buffer).replace(/\r\n?/g, '\n')

    let boundary
    while ((boundary = text.indexOf('\n\n')) !== -1) {
      const frame = text.slice(0, boundary)
      text = text.slice(boundary + 2)
      const parsed = parseFrame(frame)
      if (parsed) yield parsed
    }
    buffer = heldCR ? `${text}\r` : text
  }
  // Per the SSE spec an unterminated frame at end of stream is discarded.
}

/** Convenience: open + read in one generator (pre-stream errors are thrown on first `next()`). */
export async function* postSSE(url, opts = {}) {
  const res = await openEventStream(url, opts)
  yield* readEventStream(res, opts)
}

/** Parses one frame (already LF-normalised, without the blank-line terminator). */
export function parseFrame(frame) {
  let event = 'message'
  const dataLines = []
  for (const line of frame.split('\n')) {
    if (line === '' || line.startsWith(':')) continue // keep-alive ping / comment
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') event = value || 'message'
    else if (field === 'data') dataLines.push(value)
    // `id` and `retry` are irrelevant for a one-shot POST stream.
  }
  if (dataLines.length === 0) return null
  const raw = dataLines.join('\n')
  let data
  try {
    data = JSON.parse(raw)
  } catch {
    data = raw
  }
  return { event, data, raw }
}

function readWithTimeout(reader, ms) {
  if (!(ms > 0)) return reader.read()
  let timer
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new StreamIdleTimeoutError(ms)), ms)
  })
  return Promise.race([reader.read(), timeout]).finally(() => clearTimeout(timer))
}

async function* decodeText(body, idleTimeoutMs = 0) {
  if (typeof TextDecoderStream === 'function') {
    const reader = body.pipeThrough(new TextDecoderStream()).getReader()
    try {
      for (;;) {
        const { value, done } = await readWithTimeout(reader, idleTimeoutMs)
        if (done) return
        if (value) yield value
      }
    } finally {
      reader.cancel().catch(() => {})
    }
  } else {
    const reader = body.getReader()
    const decoder = new TextDecoder()
    try {
      for (;;) {
        const { value, done } = await readWithTimeout(reader, idleTimeoutMs)
        if (done) {
          const tail = decoder.decode()
          if (tail) yield tail
          return
        }
        yield decoder.decode(value, { stream: true })
      }
    } finally {
      reader.cancel().catch(() => {})
    }
  }
}

function abortError(signal) {
  if (signal.reason instanceof Error && signal.reason.name === 'AbortError') return signal.reason
  return new DOMException('The operation was aborted.', 'AbortError')
}

/**
 * RFC 4122 v4 UUID. `crypto.randomUUID` only exists in secure contexts
 * (https / localhost), so plain-http LAN access falls back to getRandomValues.
 */
export function uuid() {
  const c = globalThis.crypto
  if (c && typeof c.randomUUID === 'function') {
    try {
      return c.randomUUID()
    } catch {
      // fall through
    }
  }
  const bytes = new Uint8Array(16)
  if (c && typeof c.getRandomValues === 'function') {
    c.getRandomValues(bytes)
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256)
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40 // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80 // variant 10xx
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}
