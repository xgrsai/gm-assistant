import { useCallback, useEffect, useState } from 'react'
import { Loader2, Menu, UserRound } from 'lucide-react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import MessageInput from './components/MessageInput'
import ErrorBanner from './components/ErrorBanner'
import InvestigatorPanel from './components/InvestigatorPanel'
import NewCampaignDialog from './components/NewCampaignDialog'
import EmptyState from './components/EmptyState'
import RagBadge from './components/RagBadge'
import { useCampaigns } from './hooks/useCampaigns'
import { useMeta } from './hooks/useMeta'
import { useTurnStream } from './hooks/useTurnStream'

export default function App() {
  const campaigns = useCampaigns()
  const meta = useMeta()
  const { selected, selectedId, refresh, select } = campaigns
  const turn = useTurnStream(selectedId, { onCampaignMissing: () => refresh() })

  const [dialogOpen, setDialogOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sheetOpen, setSheetOpen] = useState(false)

  // Close drawers with Escape.
  useEffect(() => {
    if (!sidebarOpen && !sheetOpen) return
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setSidebarOpen(false)
        setSheetOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [sidebarOpen, sheetOpen])

  const openNew = useCallback(() => {
    setSidebarOpen(false)
    setDialogOpen(true)
  }, [])

  const selectCampaign = useCallback(
    (id) => {
      select(id)
      setSidebarOpen(false)
    },
    [select],
  )

  const noCampaigns = campaigns.status !== 'loading' && campaigns.campaigns.length === 0
  const inputHint =
    turn.phase === 'streaming'
      ? 'Хранитель відповідає — можна вже писати наступну дію'
      : turn.phase === 'failed' && turn.error?.turn
        ? 'Спершу повторіть або відкиньте перерваний хід'
        : ''

  return (
    <div className="fixed inset-0 flex overflow-hidden">
      <Sidebar
        campaigns={campaigns.campaigns}
        status={campaigns.status}
        selectedId={selectedId}
        onSelect={selectCampaign}
        onDelete={campaigns.remove}
        onNew={openNew}
        onReload={() => refresh()}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex shrink-0 items-center gap-3 border-b border-ash-800 px-3 py-3 sm:px-6">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Відкрити список кампаній"
            aria-controls="campaign-sidebar"
            aria-expanded={sidebarOpen}
            className="rounded p-1.5 text-ash-300 hover:bg-ash-800 hover:text-parchment md:hidden"
          >
            <Menu size={20} aria-hidden />
          </button>

          <div className="min-w-0 flex-1">
            {selected ? (
              <>
                <h1 className="truncate font-display text-xl font-semibold leading-tight tracking-wide text-parchment">
                  {selected.title}
                </h1>
                <p className="truncate text-xs text-ash-500">
                  {selected.scenario_title}
                  {selected.use_rag && <RagBadge className="ml-2 align-[1px]" />}
                  {selected.language && <span className="ml-2 uppercase tracking-widest text-ash-600">{selected.language}</span>}
                </p>
              </>
            ) : (
              <>
                <h1 className="font-display text-xl font-semibold leading-tight tracking-wide text-parchment">Хранитель</h1>
                <p className="text-xs text-ash-500">Ведучий для «Поклику Ктулху» 7-ї редакції</p>
              </>
            )}
          </div>

          {selected && (
            <button
              type="button"
              onClick={() => setSheetOpen(true)}
              aria-label="Відкрити аркуш слідчого"
              aria-controls="investigator-panel"
              aria-expanded={sheetOpen}
              className="inline-flex items-center gap-1.5 rounded border border-ash-700 px-2 py-1.5 text-sm text-ash-300 hover:border-ash-500 hover:text-parchment xl:hidden"
            >
              <UserRound size={16} aria-hidden />
              <span className="hidden sm:inline">Слідчий</span>
            </button>
          )}
        </header>

        {campaigns.status === 'loading' && campaigns.campaigns.length === 0 ? (
          <div className="flex flex-1 items-center justify-center gap-2 text-ash-500" role="status">
            <Loader2 size={16} className="animate-spin" aria-hidden /> Завантаження кампаній…
          </div>
        ) : noCampaigns || !selected ? (
          <EmptyState status={campaigns.status} onNew={openNew} onReload={() => refresh()} />
        ) : (
          <div className="flex min-h-0 flex-1">
            <section className="flex min-w-0 flex-1 flex-col" aria-label="Гра">
              <ChatWindow
                campaignId={selectedId}
                messages={turn.messages}
                loading={turn.loading}
                loadError={turn.loadError}
                onReload={turn.reload}
                phase={turn.phase}
                stage={turn.stage}
                liveRolls={turn.liveRolls}
                liveText={turn.liveText}
              />
              <ErrorBanner
                error={turn.error}
                busy={turn.phase === 'streaming' || turn.phase === 'discarding'}
                onRetry={turn.retry}
                onDiscard={turn.discard}
                onDismiss={turn.dismissError}
              />
              <MessageInput onSend={turn.send} canSend={turn.canSend} hint={inputHint} />
            </section>
            <InvestigatorPanel investigator={turn.investigator} open={sheetOpen} onClose={() => setSheetOpen(false)} />
          </div>
        )}
      </main>

      <NewCampaignDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onCreate={campaigns.create}
        meta={meta.meta}
        metaStatus={meta.status}
        onReloadMeta={meta.reload}
      />
    </div>
  )
}
