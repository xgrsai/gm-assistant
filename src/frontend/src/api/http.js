// Small JSON client for the backend. All URLs are relative (`/api/...`):
// in development the Vite dev server proxies them to the FastAPI backend.

export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {{status?: number, detail?: any, retryable?: boolean}} [info]
   */
  constructor(message, { status = 0, detail = null, retryable } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    // Normalised so `err.detail.code` is always readable.
    this.detail = normaliseDetail(status, detail)
    this.code = this.detail.code
    this.retryable = retryable ?? Boolean(this.detail.retryable)
  }
}

function normaliseDetail(status, detail) {
  if (detail && typeof detail === 'object' && !Array.isArray(detail) && detail.code) {
    return detail
  }
  // FastAPI validation errors: 422 with `detail: [...]`.
  if (status === 422) return { code: 'validation_error', retryable: false, errors: detail }
  if (status === 0) return { code: 'network_error', retryable: false }
  // e.g. the Vite proxy answering 500/502/504 with an empty body because the backend is down
  if (status >= 500 && (detail == null || typeof detail !== 'object')) {
    return { code: 'server_unavailable', retryable: false, raw: detail }
  }
  return { code: `http_${status}`, retryable: false, raw: detail }
}

/** Builds an ApiError from a non-2xx Response (body `{"detail": ...}`). */
export async function errorFromResponse(res) {
  let body = null
  try {
    body = await res.json()
  } catch {
    // Non-JSON error body (e.g. a proxy error page) — keep `detail` null.
  }
  const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : body
  return new ApiError(`HTTP ${res.status}`, { status: res.status, detail })
}

/** Wraps a fetch() rejection (network failure) unless it is an abort. */
export function networkError(err) {
  if (err?.name === 'AbortError') return err
  const wrapped = new ApiError(err?.message || 'Network error', { status: 0 })
  wrapped.cause = err
  return wrapped
}

/**
 * @param {string} path  e.g. `/api/campaigns`
 * @param {{method?: string, body?: any, signal?: AbortSignal}} [opts]
 */
export async function request(path, { method = 'GET', body, signal } = {}) {
  let res
  try {
    res = await fetch(path, {
      method,
      signal,
      headers: {
        Accept: 'application/json',
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch (err) {
    throw networkError(err)
  }
  if (!res.ok) throw await errorFromResponse(res)
  if (res.status === 204) return null
  const text = await res.text()
  return text ? JSON.parse(text) : null
}

const enc = encodeURIComponent

export const api = {
  listScenarios: (opts) => request('/api/scenarios', opts),
  listCampaigns: (opts) => request('/api/campaigns', opts),
  createCampaign: (body, opts) => request('/api/campaigns', { ...opts, method: 'POST', body }),
  getCampaign: (id, opts) => request(`/api/campaigns/${enc(id)}`, opts),
  deleteCampaign: (id, opts) => request(`/api/campaigns/${enc(id)}`, { ...opts, method: 'DELETE' }),
  getMessages: (id, opts) => request(`/api/campaigns/${enc(id)}/messages`, opts),
  discardTurn: (id, opts) =>
    request(`/api/campaigns/${enc(id)}/turns/discard`, { ...opts, method: 'POST' }),
  turnsUrl: (id) => `/api/campaigns/${enc(id)}/turns`,
  retryUrl: (id) => `/api/campaigns/${enc(id)}/turns/retry`,
}
