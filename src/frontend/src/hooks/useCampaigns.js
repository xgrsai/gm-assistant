import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/http.js'

const STORAGE_KEY = 'gm-assistant.selectedCampaignId'

function readStoredId() {
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null // storage blocked (private mode, sandboxed iframe, …)
  }
}

function writeStoredId(id) {
  try {
    if (id) window.localStorage.setItem(STORAGE_KEY, id)
    else window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore
  }
}

/**
 * Campaign registry: list / create / delete, plus the selected campaign id
 * (persisted in localStorage). Campaigns are never modified client-side
 * beyond what the server returns.
 */
export function useCampaigns() {
  const [campaigns, setCampaigns] = useState([])
  const [status, setStatus] = useState('loading') // loading | ready | error
  const [error, setError] = useState(null)
  const [selectedId, setSelectedId] = useState(readStoredId)
  const campaignsRef = useRef(campaigns)
  campaignsRef.current = campaigns

  const refresh = useCallback(async (signal) => {
    setStatus((s) => (s === 'ready' ? s : 'loading'))
    try {
      const data = await api.listCampaigns({ signal })
      const items = Array.isArray(data?.items) ? data.items : []
      setCampaigns(items)
      setError(null)
      setStatus('ready')
      setSelectedId((prev) => (items.some((c) => c.id === prev) ? prev : (items[0]?.id ?? null)))
    } catch (err) {
      if (err?.name === 'AbortError') return
      setError(err)
      setStatus('error')
    }
  }, [])

  useEffect(() => {
    const ac = new AbortController()
    refresh(ac.signal)
    return () => ac.abort()
  }, [refresh])

  useEffect(() => {
    // Only persist once the list is known, so a stale id is not written back.
    if (status === 'ready') writeStoredId(selectedId)
  }, [selectedId, status])

  const select = useCallback((id) => setSelectedId(id), [])

  /** Creates a campaign on the server and selects it. Throws ApiError on failure. */
  const create = useCallback(async ({ title, scenario_id, language }) => {
    const campaign = await api.createCampaign({ title, scenario_id, language })
    setCampaigns((prev) => [campaign, ...prev.filter((c) => c.id !== campaign.id)])
    setSelectedId(campaign.id)
    setStatus('ready')
    return campaign
  }, [])

  /** Deletes a campaign; if it was selected, selects the next one. */
  const remove = useCallback(async (id) => {
    try {
      await api.deleteCampaign(id)
    } catch (err) {
      if (err?.status !== 404) throw err // already gone → treat as deleted
    }
    const next = campaignsRef.current.filter((c) => c.id !== id)
    setCampaigns(next)
    setSelectedId((sel) => (sel === id ? (next[0]?.id ?? null) : sel))
  }, [])

  const selected = campaigns.find((c) => c.id === selectedId) ?? null

  return { campaigns, status, error, selectedId, selected, select, create, remove, refresh }
}
