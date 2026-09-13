import { useState } from 'react'
import { Eye, Loader2, Plus, RotateCcw, ScrollText, Trash2, X } from 'lucide-react'
import RagBadge from './RagBadge'

function CampaignItem({ campaign, active, onSelect, onDelete }) {
  const [confirming, setConfirming] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [failed, setFailed] = useState(false)

  const confirmDelete = async () => {
    setDeleting(true)
    setFailed(false)
    try {
      await onDelete(campaign.id)
    } catch {
      setFailed(true)
      setDeleting(false)
      setConfirming(false)
    }
  }

  if (confirming) {
    return (
      <li className="rounded border border-blood/50 bg-blood/10 px-3 py-2 text-sm">
        <p className="text-parchment">
          Видалити «<span className="break-words">{campaign.title}</span>»?
        </p>
        <p className="mt-0.5 text-xs text-ash-400">Історію кампанії буде втрачено.</p>
        <div className="mt-2 flex gap-2">
          <button
            type="button"
            onClick={confirmDelete}
            disabled={deleting}
            className="inline-flex items-center gap-1 rounded border border-blood/70 bg-blood/30 px-2 py-1 text-xs text-parchment hover:bg-blood/50 disabled:opacity-50"
          >
            {deleting ? <Loader2 size={12} className="animate-spin" aria-hidden /> : <Trash2 size={12} aria-hidden />}
            Видалити
          </button>
          <button
            type="button"
            onClick={() => setConfirming(false)}
            disabled={deleting}
            className="rounded border border-ash-600 px-2 py-1 text-xs text-ash-200 hover:text-parchment disabled:opacity-50"
          >
            Скасувати
          </button>
        </div>
      </li>
    )
  }

  return (
    <li className="group relative">
      <button
        type="button"
        onClick={() => onSelect(campaign.id)}
        aria-current={active ? 'true' : undefined}
        className={`flex w-full items-start gap-2 rounded border-l-2 py-2 pl-3 pr-9 text-left transition-colors
          ${active ? 'border-ember bg-ash-800 text-parchment' : 'border-transparent text-ash-400 hover:bg-ash-800/50 hover:text-ash-200'}`}
      >
        <ScrollText size={14} className={`mt-1 shrink-0 ${active ? 'text-ember' : 'text-ember/60'}`} aria-hidden />
        <span className="min-w-0">
          <span className="flex min-w-0 items-center gap-1.5">
            <span className="truncate text-sm">{campaign.title}</span>
            {campaign.use_rag && <RagBadge />}
          </span>
          {campaign.scenario_title && (
            <span className="block truncate text-xs text-ash-500">{campaign.scenario_title}</span>
          )}
          {failed && <span className="block text-xs text-red-300">Не вдалося видалити</span>}
        </span>
      </button>
      <button
        type="button"
        onClick={() => setConfirming(true)}
        aria-label={`Видалити кампанію «${campaign.title}»`}
        title="Видалити"
        className="absolute right-1.5 top-2 rounded p-1 text-ash-500 opacity-100 transition hover:bg-blood/30 hover:text-red-200 focus-visible:opacity-100 md:opacity-0 md:group-hover:opacity-100"
      >
        <Trash2 size={13} aria-hidden />
      </button>
    </li>
  )
}

/**
 * Campaign list. Static column on `md+`, off-canvas drawer below.
 */
export default function Sidebar({ campaigns, status, selectedId, onSelect, onDelete, onNew, onReload, open, onClose }) {
  return (
    <>
      <div
        className={`fixed inset-0 z-30 bg-black/60 transition-opacity md:hidden ${open ? 'opacity-100' : 'pointer-events-none opacity-0'}`}
        onClick={onClose}
        aria-hidden
      />
      <aside
        id="campaign-sidebar"
        aria-label="Кампанії"
        className={`fixed inset-y-0 left-0 z-40 flex w-[min(17rem,85vw)] flex-col border-r border-ash-800 bg-ash-900 transition-[transform,visibility] duration-300
          md:visible md:static md:z-auto md:w-64 md:translate-x-0 md:bg-ash-900/60 md:transition-none
          ${open ? 'visible translate-x-0' : 'invisible -translate-x-full'}`}
      >
        <div className="flex items-center justify-between gap-2 border-b border-ash-800 px-4 py-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <Eye size={18} className="shrink-0 text-ember" aria-hidden />
            <div className="min-w-0">
              <p className="font-display text-lg font-semibold leading-none tracking-wide text-parchment">Хранитель</p>
              <p className="mt-1 truncate text-[11px] uppercase tracking-[0.18em] text-ash-500">Поклик Ктулху · 7e</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Закрити список кампаній"
            className="rounded p-1 text-ash-400 hover:bg-ash-800 hover:text-parchment md:hidden"
          >
            <X size={16} aria-hidden />
          </button>
        </div>

        <nav className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-2 py-4" aria-label="Список кампаній">
          <p className="mb-2 px-2 text-xs uppercase tracking-widest text-ash-500">Кампанії</p>

          {status === 'loading' && campaigns.length === 0 && (
            <p className="flex items-center gap-2 px-2 py-2 text-sm text-ash-500" role="status">
              <Loader2 size={14} className="animate-spin" aria-hidden /> Завантаження…
            </p>
          )}

          {status === 'error' && (
            <div className="px-2 py-2 text-sm text-ash-400">
              <p>Сервер недоступний.</p>
              <button
                type="button"
                onClick={onReload}
                className="mt-2 inline-flex items-center gap-1.5 rounded border border-ash-700 px-2 py-1 text-xs text-parchment hover:border-ember/60"
              >
                <RotateCcw size={12} aria-hidden /> Оновити
              </button>
            </div>
          )}

          {status === 'ready' && campaigns.length === 0 && (
            <p className="px-2 py-2 text-sm italic text-ash-500">Жодної справи ще не відкрито.</p>
          )}

          <ul className="space-y-1">
            {campaigns.map((c) => (
              <CampaignItem
                key={c.id}
                campaign={c}
                active={c.id === selectedId}
                onSelect={onSelect}
                onDelete={onDelete}
              />
            ))}
          </ul>
        </nav>

        <div className="border-t border-ash-800 px-2 py-3">
          <button
            type="button"
            onClick={onNew}
            className="flex w-full items-center gap-2 rounded px-3 py-2 text-sm text-ash-300 transition-colors hover:bg-ash-800/60 hover:text-parchment"
          >
            <Plus size={15} className="text-ember" aria-hidden />
            Нова кампанія
          </button>
        </div>
      </aside>
    </>
  )
}
