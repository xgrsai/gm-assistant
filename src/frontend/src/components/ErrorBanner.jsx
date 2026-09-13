import { AlertTriangle, RotateCcw, Trash2, X } from 'lucide-react'
import { errorMessage } from '../i18n/uk.js'

/**
 * @param {{error: {code: string, retryable: boolean, turn: boolean}|null,
 *          busy?: boolean, onRetry: () => void, onDiscard: () => void, onDismiss: () => void}} props
 */
export default function ErrorBanner({ error, busy = false, onRetry, onDiscard, onDismiss }) {
  if (!error) return null
  const turnFailed = Boolean(error.turn)

  return (
    <div className="shrink-0 px-3 pt-3 sm:px-6">
      <div
        role="alert"
        className={`mx-auto flex max-w-3xl flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border px-4 py-3 text-sm
          ${turnFailed ? 'border-ember/50 bg-ember/10' : 'border-blood/50 bg-blood/15'}`}
      >
        <AlertTriangle size={16} className="shrink-0 self-start text-ember sm:mt-0.5" aria-hidden />
        <div className="min-w-0 flex-1 basis-56">
          <p className="text-parchment">{errorMessage(error.code)}</p>
          {turnFailed && error.code !== 'turn_incomplete' && (
            <p className="mt-0.5 text-xs text-ash-400">
              {error.retryable
                ? 'Хід можна повторити (уже кинуті кубики не перекидаються) або відкинути.'
                : 'Хід лишився незавершеним. Відкиньте його або спробуйте повторити.'}
            </p>
          )}
          {error.code && <p className="mt-0.5 font-mono text-[11px] text-ash-600">{error.code}</p>}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          {turnFailed ? (
            <>
              <button
                type="button"
                onClick={onRetry}
                disabled={busy}
                className="inline-flex items-center gap-1.5 rounded border border-ember/60 bg-ember/20 px-3 py-1.5 text-parchment transition-colors hover:bg-ember/35 disabled:cursor-wait disabled:opacity-50"
              >
                <RotateCcw size={14} aria-hidden />
                Повторити
              </button>
              <button
                type="button"
                onClick={onDiscard}
                disabled={busy}
                className="inline-flex items-center gap-1.5 rounded border border-ash-600 px-3 py-1.5 text-ash-200 transition-colors hover:border-ash-400 hover:text-parchment disabled:cursor-wait disabled:opacity-50"
              >
                <Trash2 size={14} aria-hidden />
                Відкинути
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={onDismiss}
              aria-label="Закрити повідомлення"
              className="rounded p-1.5 text-ash-400 transition-colors hover:bg-ash-800 hover:text-parchment"
            >
              <X size={16} aria-hidden />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
