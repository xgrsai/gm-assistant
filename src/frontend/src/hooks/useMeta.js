import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/http.js'

/**
 * Server metadata (`GET /api/meta`): version, experiment arms and the default arm.
 * Loaded once on mount; `reload()` refetches (e.g. after an `unknown_arm` error
 * or when the first attempt failed because the backend was down).
 */
export function useMeta() {
  const [meta, setMeta] = useState(null)
  const [status, setStatus] = useState('loading') // loading | ready | error
  const ctlRef = useRef(null)

  const reload = useCallback(async () => {
    ctlRef.current?.abort()
    const controller = new AbortController()
    ctlRef.current = controller
    setStatus('loading')
    try {
      const data = await api.getMeta({ signal: controller.signal })
      if (controller.signal.aborted) return
      setMeta({
        ...data,
        arms: Array.isArray(data?.arms) ? data.arms.filter((a) => a && typeof a.id === 'string') : [],
      })
      setStatus('ready')
    } catch (err) {
      if (err?.name === 'AbortError' || controller.signal.aborted) return
      setStatus('error')
    }
  }, [])

  useEffect(() => {
    reload()
    return () => ctlRef.current?.abort()
  }, [reload])

  return { meta, status, reload }
}
