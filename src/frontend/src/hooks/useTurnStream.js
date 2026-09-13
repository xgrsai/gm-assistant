import { useCallback, useEffect, useReducer, useRef } from 'react'
import { api } from '../api/http.js'
import { openEventStream, readEventStream, uuid } from '../api/sse.js'

/*
 * Conversation state of the active campaign.
 *
 *  phase: 'idle'       nothing in flight
 *         'streaming'  a turn (or a retry) is streaming
 *         'failed'     the last turn stopped; `error` explains why
 *         'discarding' POST /turns/discard in flight
 *
 *  error: { code, retryable, message, turn }
 *         turn === true  → the server holds an incomplete turn: show Retry / Discard
 *                          (regardless of `retryable`; the backend keeps the turn incomplete)
 *         turn === false → a plain notice, dismissible
 */

function initialState(cid) {
  return {
    cid,
    loading: Boolean(cid),
    loadError: null,
    loaded: false,
    messages: [],
    investigator: null,
    phase: 'idle',
    stage: null,
    liveRolls: [],
    liveText: '',
    dedupeRolls: false,
    error: null,
  }
}

// fastapi.sse sends a keep-alive ping every 15 s, so 45 s of silence means a dead connection.
const STREAM_IDLE_TIMEOUT_MS = 45_000

const TURN_ERROR_INCOMPLETE = { code: 'turn_incomplete', retryable: true, message: null, turn: true }

function rollKey(r) {
  return [r?.turn_no, r?.kind, r?.target, r?.purpose, r?.roll, r?.tens?.join('.'), r?.units, r?.pushed].join('|')
}

function reducer(state, action) {
  if (action.type === 'reset') return initialState(action.cid)
  if (action.cid !== state.cid) return state // event for a campaign that is no longer active

  switch (action.type) {
    case 'loadStart':
      return { ...state, loading: true, loadError: null }

    case 'loaded': {
      const { items, incomplete, investigator } = action.data ?? {}
      const next = {
        ...state,
        loading: false,
        loaded: true,
        loadError: null,
        messages: Array.isArray(items) ? items : [],
        investigator: investigator ?? state.investigator,
      }
      if (state.phase === 'streaming') return next // never clobber a live turn
      if (incomplete) {
        return {
          ...next,
          phase: 'failed',
          stage: null,
          liveText: '',
          error: state.error?.turn ? state.error : TURN_ERROR_INCOMPLETE,
        }
      }
      return {
        ...next,
        phase: 'idle',
        stage: null,
        liveRolls: [],
        liveText: '',
        error: state.error?.turn ? null : state.error,
      }
    }

    case 'loadFailed':
      return { ...state, loading: false, loadError: action.error }

    case 'sendStart':
      return {
        ...state,
        messages: [...state.messages, action.message],
        phase: 'streaming',
        stage: null,
        liveRolls: [],
        liveText: '',
        dedupeRolls: false,
        error: null,
      }

    case 'sendRejected':
      return {
        ...state,
        messages: state.messages.filter((m) => m.id !== action.localId),
        phase: 'idle',
        stage: null,
      }

    case 'retryStart':
      // Narration restarts from its checkpoint, so the partial text is dropped;
      // dice already rolled are kept (and de-duplicated if re-emitted).
      return { ...state, phase: 'streaming', stage: null, liveText: '', dedupeRolls: true, error: null }

    case 'progress':
      return { ...state, stage: action.node ?? state.stage }

    case 'roll': {
      if (state.dedupeRolls) {
        const key = rollKey(action.roll)
        if (state.liveRolls.some((r) => rollKey(r) === key)) return state
      }
      return { ...state, liveRolls: [...state.liveRolls, action.roll] }
    }

    case 'token':
      return { ...state, liveText: state.liveText + action.text }

    case 'done': {
      const raw = action.message ?? {}
      const message = {
        ...raw,
        id: raw.id ?? `keeper-${uuid()}`,
        role: raw.role ?? 'keeper',
        text: typeof raw.text === 'string' ? raw.text : state.liveText,
        rolls: Array.isArray(raw.rolls) ? raw.rolls : state.liveRolls,
        fromStream: true, // already shown live: render without the entry animation
      }
      const messages = state.messages
        .filter((m) => m.id !== message.id)
        .map((m) => (m.pending ? { ...m, pending: false } : m))
      return {
        ...state,
        messages: [...messages, message],
        investigator: action.investigator ?? state.investigator,
        phase: 'idle',
        stage: null,
        liveRolls: [],
        liveText: '',
        dedupeRolls: false,
        error: null,
      }
    }

    case 'failed':
      return { ...state, phase: 'failed', stage: null, error: action.error }

    case 'notice':
      return { ...state, error: action.error }

    case 'discardStart':
      return { ...state, phase: 'discarding' }

    case 'discarded':
      return { ...state, phase: 'idle', stage: null, liveRolls: [], liveText: '', dedupeRolls: false, error: null }

    case 'dismiss':
      return { ...state, error: null, phase: state.phase === 'failed' ? 'idle' : state.phase }

    default:
      return state
  }
}

function toError(err, overrides = {}) {
  return {
    code: err?.code ?? err?.detail?.code ?? 'internal_error',
    retryable: Boolean(err?.retryable ?? err?.detail?.retryable),
    message: err?.detail?.message ?? null,
    status: err?.status,
    turn: false,
    ...overrides,
  }
}

/**
 * @param {string|null} campaignId  the active campaign
 * @param {{onCampaignMissing?: (id: string) => void}} [options]
 */
export function useTurnStream(campaignId, { onCampaignMissing } = {}) {
  const [rawState, dispatch] = useReducer(reducer, campaignId, initialState)
  const state = rawState.cid === campaignId ? rawState : initialState(campaignId)

  const activeRef = useRef(campaignId) // campaign whose events may be applied
  const ctlRef = useRef(null) // { cid, controller } — one AbortController per campaign
  const busyRef = useRef(null) // op token of the send/retry/discard in flight
  const opSeq = useRef(0)
  const tokenBuf = useRef({ cid: null, text: '', frame: 0 })
  const missingRef = useRef(onCampaignMissing)
  missingRef.current = onCampaignMissing

  const isLive = (cid, signal) => !signal.aborted && activeRef.current === cid

  // Tokens are coalesced per animation frame to avoid one render per chunk.
  const flushTokens = useCallback(() => {
    const buf = tokenBuf.current
    if (buf.frame) cancelAnimationFrame(buf.frame)
    buf.frame = 0
    if (buf.text && buf.cid === activeRef.current) dispatch({ type: 'token', cid: buf.cid, text: buf.text })
    buf.text = ''
  }, [])

  const pushToken = useCallback(
    (cid, text) => {
      const buf = tokenBuf.current
      if (buf.cid !== cid) {
        buf.cid = cid
        buf.text = ''
      }
      buf.text += text
      if (!buf.frame) buf.frame = requestAnimationFrame(flushTokens)
    },
    [flushTokens],
  )

  const load = useCallback(async (cid, signal) => {
    dispatch({ type: 'loadStart', cid })
    try {
      const data = await api.getMessages(cid, { signal })
      if (!isLive(cid, signal)) return
      dispatch({ type: 'loaded', cid, data })
    } catch (err) {
      if (err?.name === 'AbortError' || !isLive(cid, signal)) return
      dispatch({ type: 'loadFailed', cid, error: toError(err) })
      if (err?.status === 404) missingRef.current?.(cid)
    }
  }, [])

  useEffect(() => {
    activeRef.current = campaignId
    busyRef.current = null
    const controller = new AbortController()
    ctlRef.current = { cid: campaignId, controller }
    tokenBuf.current = { cid: campaignId, text: '', frame: 0 }
    dispatch({ type: 'reset', cid: campaignId })
    if (campaignId) load(campaignId, controller.signal)
    return () => {
      controller.abort()
      const buf = tokenBuf.current
      if (buf.frame) cancelAnimationFrame(buf.frame)
      buf.frame = 0
      buf.text = ''
    }
  }, [campaignId, load])

  /** Reads an opened turn stream to its terminal `done` / `error` event. */
  const consume = useCallback(
    async (cid, signal, response) => {
      let terminal = false
      try {
        for await (const { event, data } of readEventStream(response, { signal, idleTimeoutMs: STREAM_IDLE_TIMEOUT_MS })) {
          if (!isLive(cid, signal)) return
          switch (event) {
            case 'progress':
              flushTokens()
              dispatch({ type: 'progress', cid, node: data?.node })
              break
            case 'roll':
              flushTokens()
              dispatch({ type: 'roll', cid, roll: data })
              break
            case 'token':
              if (typeof data?.text === 'string' && data.text) pushToken(cid, data.text)
              break
            case 'done':
              terminal = true
              flushTokens()
              dispatch({ type: 'done', cid, message: data?.message, investigator: data?.investigator })
              break
            case 'error': {
              terminal = true
              flushTokens()
              const code = data?.code ?? 'internal_error'
              const retryable = Boolean(data?.retryable)
              if (code === 'turn_in_progress') {
                // Another stream owns the campaign — resync instead of claiming a failed turn.
                dispatch({ type: 'discarded', cid })
                dispatch({ type: 'notice', cid, error: { code, retryable, message: data?.message ?? null, turn: false } })
                load(cid, signal)
              } else {
                // Any in-stream error leaves the turn incomplete on the server
                // (GET /messages → incomplete: true), so Retry / Discard are always offered.
                dispatch({
                  type: 'failed',
                  cid,
                  error: { code, retryable, message: data?.message ?? null, turn: true },
                })
              }
              break
            }
            default:
              break // unknown events are ignored for forward compatibility
          }
          if (terminal) break // exactly one of done / error ends a stream
        }
        if (!terminal && isLive(cid, signal)) {
          flushTokens()
          dispatch({ type: 'failed', cid, error: { code: 'connection_lost', retryable: true, message: null, turn: true } })
        }
      } catch (err) {
        if (err?.name === 'AbortError' || !isLive(cid, signal)) return
        flushTokens()
        dispatch({ type: 'failed', cid, error: { code: 'connection_lost', retryable: true, message: null, turn: true } })
      }
    },
    [flushTokens, pushToken, load],
  )

  /** Handles an ordinary HTTP error that happened before a stream started. */
  const handlePreStreamError = useCallback(
    (cid, signal, err, context) => {
      const error = toError(err)
      switch (error.code) {
        case 'turn_incomplete':
          dispatch({ type: 'failed', cid, error: { ...error, retryable: true, turn: true } })
          load(cid, signal)
          break
        case 'not_found':
          dispatch({ type: context === 'send' ? 'notice' : 'failed', cid, error })
          missingRef.current?.(cid)
          break
        case 'duplicate_turn':
        case 'nothing_to_retry':
          dispatch({ type: 'discarded', cid })
          dispatch({ type: 'notice', cid, error })
          load(cid, signal)
          break
        default:
          if (context === 'retry') {
            // The incomplete turn is still on the server: keep Retry / Discard available.
            dispatch({ type: 'failed', cid, error: { ...error, retryable: true, turn: true } })
          } else {
            dispatch({ type: 'notice', cid, error })
          }
      }
    },
    [load],
  )

  const begin = () => {
    const cid = activeRef.current
    const ctl = ctlRef.current
    if (!cid || !ctl || ctl.cid !== cid || ctl.controller.signal.aborted || busyRef.current) return null
    const op = ++opSeq.current
    busyRef.current = op
    return { cid, signal: ctl.controller.signal, end: () => busyRef.current === op && (busyRef.current = null) }
  }

  /**
   * Sends a player action. Resolves `true` once the server accepted the stream
   * (the stream keeps running in the background), `false` if it was refused
   * before streaming — so the caller can keep the typed text.
   */
  const send = useCallback(
    async (text) => {
      const trimmed = String(text ?? '').trim()
      if (!trimmed) return false
      const op = begin()
      if (!op) return false
      const { cid, signal } = op
      const clientTurnId = uuid()
      const localId = `local-${clientTurnId}`
      dispatch({ type: 'sendStart', cid, message: { id: localId, role: 'player', text: trimmed, pending: true } })

      let response
      try {
        response = await openEventStream(api.turnsUrl(cid), {
          body: { client_turn_id: clientTurnId, text: trimmed },
          signal,
        })
      } catch (err) {
        op.end()
        if (err?.name === 'AbortError' || !isLive(cid, signal)) return false
        dispatch({ type: 'sendRejected', cid, localId })
        handlePreStreamError(cid, signal, err, 'send')
        return false
      }
      consume(cid, signal, response).finally(op.end)
      return true
    },
    [consume, handlePreStreamError],
  )

  /** POST /turns/retry — resumes the incomplete turn, same stream handling. */
  const retry = useCallback(async () => {
    const op = begin()
    if (!op) return
    const { cid, signal } = op
    dispatch({ type: 'retryStart', cid })
    try {
      const response = await openEventStream(api.retryUrl(cid), { signal })
      await consume(cid, signal, response)
    } catch (err) {
      if (err?.name === 'AbortError' || !isLive(cid, signal)) return
      handlePreStreamError(cid, signal, err, 'retry')
    } finally {
      op.end()
    }
  }, [consume, handlePreStreamError])

  /** POST /turns/discard, then reload the transcript from the server. */
  const discard = useCallback(async () => {
    const op = begin()
    if (!op) return
    const { cid, signal } = op
    dispatch({ type: 'discardStart', cid })
    try {
      await api.discardTurn(cid, { signal })
    } catch (err) {
      if (err?.name === 'AbortError' || !isLive(cid, signal)) return op.end()
      if (err?.code !== 'nothing_to_retry') {
        dispatch({ type: 'failed', cid, error: { ...toError(err), retryable: true, turn: true } })
        return op.end()
      }
    }
    if (!isLive(cid, signal)) return op.end()
    dispatch({ type: 'discarded', cid })
    await load(cid, signal)
    op.end()
  }, [load])

  /** Hides a notice. After a non-retryable failure, resync with the server. */
  const dismissError = useCallback(() => {
    const cid = activeRef.current
    const ctl = ctlRef.current
    if (!cid || !ctl) return
    const wasFailed = rawState.phase === 'failed'
    dispatch({ type: 'dismiss', cid })
    if (wasFailed && !busyRef.current) load(cid, ctl.controller.signal)
  }, [rawState.phase, load])

  const reload = useCallback(() => {
    const cid = activeRef.current
    const ctl = ctlRef.current
    if (cid && ctl && !busyRef.current) load(cid, ctl.controller.signal)
  }, [load])

  const isStreaming = state.phase === 'streaming'
  const blocked = state.phase !== 'idle' || state.loading || !state.loaded

  return {
    ...state,
    isStreaming,
    canSend: Boolean(campaignId) && !blocked,
    send,
    retry,
    discard,
    dismissError,
    reload,
  }
}
