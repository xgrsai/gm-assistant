import { Eye, Plus, RotateCcw } from 'lucide-react'

export default function EmptyState({ status, onNew, onReload }) {
  const unavailable = status === 'error'
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-6 py-10">
      <div className="max-w-md text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full border border-ember/30 bg-ember/10">
          <Eye size={26} className="animate-flicker text-ember" aria-hidden />
        </div>
        {unavailable ? (
          <>
            <h2 className="mt-5 font-display text-3xl font-semibold text-parchment">Архів мовчить</h2>
            <p className="mt-2 leading-relaxed text-ash-400">
              Не вдалося отримати список кампаній. Переконайтеся, що бекенд запущено, і спробуйте ще раз.
            </p>
            <button
              type="button"
              onClick={onReload}
              className="mt-6 inline-flex items-center gap-2 rounded border border-ash-600 px-4 py-2 text-parchment hover:border-ember/60 hover:bg-ember/10"
            >
              <RotateCcw size={15} aria-hidden /> Оновити
            </button>
          </>
        ) : (
          <>
            <h2 className="mt-5 font-display text-3xl font-semibold text-parchment">Жодної справи ще не відкрито</h2>
            <p className="mt-2 leading-relaxed text-ash-400">
              Туман над містом густішає, а в редакції чекає лист без зворотної адреси. Створіть кампанію — і Хранитель
              поведе вашого слідчого туди, куди розважливі люди не ходять.
            </p>
            <button
              type="button"
              onClick={onNew}
              className="mt-6 inline-flex items-center gap-2 rounded border border-ember/60 bg-ember/20 px-4 py-2 text-parchment transition-colors hover:bg-ember/35"
            >
              <Plus size={16} aria-hidden /> Нова кампанія
            </button>
          </>
        )}
      </div>
    </div>
  )
}
