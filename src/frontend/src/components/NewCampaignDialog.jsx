import { useEffect, useId, useRef, useState } from 'react'
import { Loader2, X } from 'lucide-react'
import { api } from '../api/http.js'
import { LANGUAGE_LABELS, errorMessage, localized } from '../i18n/uk.js'

const TITLE_MAX = 120

/**
 * Modal built on <dialog> (native focus handling + Esc). Creating a campaign
 * never touches existing ones.
 */
export default function NewCampaignDialog({ open, onClose, onCreate }) {
  const dialogRef = useRef(null)
  const titleRef = useRef(null)
  const ids = useId()

  const [title, setTitle] = useState('')
  const [scenarioId, setScenarioId] = useState('')
  const [language, setLanguage] = useState('uk')
  const [scenarios, setScenarios] = useState([])
  const [scenarioStatus, setScenarioStatus] = useState('idle') // idle | loading | ready | error
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState(null)

  // Sync the native dialog with `open`.
  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (open && !dialog.open) {
      setTitle('')
      setSubmitError(null)
      setSubmitting(false)
      dialog.showModal()
      requestAnimationFrame(() => titleRef.current?.focus())
    } else if (!open && dialog.open) {
      dialog.close()
    }
  }, [open])

  // Load scenarios each time the dialog opens.
  useEffect(() => {
    if (!open) return
    const ac = new AbortController()
    setScenarioStatus('loading')
    api
      .listScenarios({ signal: ac.signal })
      .then((data) => {
        const items = Array.isArray(data?.items) ? data.items : []
        setScenarios(items)
        setScenarioId((prev) => (items.some((s) => s.id === prev) ? prev : (items[0]?.id ?? '')))
        setScenarioStatus('ready')
      })
      .catch((err) => {
        if (err?.name === 'AbortError') return
        setScenarioStatus('error')
      })
    return () => ac.abort()
  }, [open])

  const scenario = scenarios.find((s) => s.id === scenarioId)
  const trimmedTitle = title.trim()
  const canSubmit = !submitting && trimmedTitle.length > 0 && trimmedTitle.length <= TITLE_MAX && Boolean(scenarioId)

  const submit = async (e) => {
    e.preventDefault()
    if (!canSubmit) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      await onCreate({ title: trimmedTitle, scenario_id: scenarioId, language })
      onClose()
    } catch (err) {
      setSubmitError(err)
    } finally {
      setSubmitting(false)
    }
  }

  const fieldClass =
    'w-full rounded border border-ash-700 bg-ash-800/70 px-3 py-2 text-parchment placeholder:text-ash-600 focus:border-ash-500 focus:outline-none'

  return (
    <dialog
      ref={dialogRef}
      onClose={onClose}
      onCancel={(e) => {
        if (submitting) e.preventDefault()
      }}
      onClick={(e) => {
        if (e.target === dialogRef.current && !submitting) onClose() // backdrop click
      }}
      aria-labelledby={`${ids}-title`}
      className="w-[min(32rem,calc(100vw-2rem))] max-w-none rounded-xl border border-ash-700 bg-ash-900 p-0 text-parchment shadow-2xl"
    >
      <form onSubmit={submit} className="flex flex-col gap-4 p-5 sm:p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id={`${ids}-title`} className="font-display text-2xl font-semibold leading-tight">
              Нова кампанія
            </h2>
            <p className="mt-1 text-sm text-ash-400">Оберіть сценарій — Хранитель відкриє справу.</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            aria-label="Закрити"
            className="rounded p-1 text-ash-400 hover:bg-ash-800 hover:text-parchment"
          >
            <X size={18} aria-hidden />
          </button>
        </div>

        <div>
          <label htmlFor={`${ids}-name`} className="mb-1 block text-xs uppercase tracking-widest text-ash-500">
            Назва
          </label>
          <input
            id={`${ids}-name`}
            ref={titleRef}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            maxLength={TITLE_MAX}
            required
            placeholder="Напр., Вечір у Львові"
            className={fieldClass}
          />
        </div>

        <div>
          <label htmlFor={`${ids}-scenario`} className="mb-1 block text-xs uppercase tracking-widest text-ash-500">
            Сценарій
          </label>
          {scenarioStatus === 'loading' && scenarios.length === 0 ? (
            <p className="flex items-center gap-2 py-2 text-sm text-ash-500" role="status">
              <Loader2 size={14} className="animate-spin" aria-hidden /> Завантаження сценаріїв…
            </p>
          ) : scenarioStatus === 'error' ? (
            <p className="py-2 text-sm text-red-300" role="alert">
              Не вдалося завантажити сценарії. {errorMessage('network_error')}
            </p>
          ) : scenarios.length === 0 ? (
            <p className="py-2 text-sm italic text-ash-500">На сервері немає встановлених сценаріїв.</p>
          ) : (
            <>
              <select
                id={`${ids}-scenario`}
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
                className={fieldClass}
              >
                {scenarios.map((s) => (
                  <option key={s.id} value={s.id}>
                    {localized(s.title, language) || s.id}
                    {localized(s.era, language) ? ` · ${localized(s.era, language)}` : ''}
                  </option>
                ))}
              </select>
              {scenario && localized(scenario.blurb, language) && (
                <p className="mt-2 rounded border border-ash-800 bg-ash-800/30 px-3 py-2 text-sm italic leading-relaxed text-ash-300">
                  {localized(scenario.blurb, language)}
                </p>
              )}
            </>
          )}
        </div>

        <div>
          <label htmlFor={`${ids}-lang`} className="mb-1 block text-xs uppercase tracking-widest text-ash-500">
            Мова гри
          </label>
          <select id={`${ids}-lang`} value={language} onChange={(e) => setLanguage(e.target.value)} className={fieldClass}>
            {Object.entries(LANGUAGE_LABELS).map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
        </div>

        {submitError && (
          <p role="alert" className="rounded border border-blood/50 bg-blood/15 px-3 py-2 text-sm text-red-200">
            Не вдалося створити кампанію. {errorMessage(submitError.code)}
          </p>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="rounded border border-ash-700 px-4 py-2 text-sm text-ash-300 hover:text-parchment disabled:opacity-50"
          >
            Скасувати
          </button>
          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex items-center gap-2 rounded border border-ember/60 bg-ember/25 px-4 py-2 text-sm text-parchment transition-colors hover:bg-ember/40 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting && <Loader2 size={14} className="animate-spin" aria-hidden />}
            Відкрити справу
          </button>
        </div>
      </form>
    </dialog>
  )
}
