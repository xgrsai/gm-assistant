import { useCallback, useLayoutEffect, useRef } from 'react'
import { Loader2 } from 'lucide-react'
import MessageBubble, { KeeperAvatar } from './MessageBubble'
import RollCard from './RollCard'
import { stageLabel } from '../i18n/uk.js'

const NEAR_BOTTOM_PX = 120

function RollList({ rolls, animate = true }) {
  if (!rolls?.length) return null
  return (
    <div className="flex flex-col gap-2 pl-10" role="group" aria-label="Кидки цього ходу">
      {rolls.map((roll, i) => (
        <RollCard key={`${roll.turn_no ?? 't'}-${i}`} roll={roll} animate={animate} />
      ))}
    </div>
  )
}

function StageIndicator({ stage }) {
  return (
    <div className="flex items-center gap-2 pl-10 text-sm italic tracking-wide text-ash-400" role="status">
      <span className="inline-block h-1.5 w-1.5 animate-flicker rounded-full bg-ember" aria-hidden />
      {stageLabel(stage)}
    </div>
  )
}

/**
 * @param {object} props
 * @param {Array} props.messages
 * @param {'idle'|'streaming'|'failed'|'discarding'} props.phase
 * @param {string|null} props.stage
 * @param {Array} props.liveRolls
 * @param {string} props.liveText
 */
export default function ChatWindow({ campaignId, messages, loading, loadError, onReload, phase, stage, liveRolls, liveText }) {
  const scrollRef = useRef(null)
  const nearBottomRef = useRef(true)
  const lastCampaignRef = useRef(null)

  const onScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    nearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX
  }, [])

  const streaming = phase === 'streaming'
  const failed = phase === 'failed'
  const showLive = streaming || (failed && (liveRolls.length > 0 || liveText))

  // Jump to the bottom when a campaign opens; afterwards follow new content
  // only if the reader is already near the bottom.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    if (lastCampaignRef.current !== campaignId && !loading) {
      lastCampaignRef.current = campaignId
      el.scrollTop = el.scrollHeight
      nearBottomRef.current = true
      return
    }
    if (nearBottomRef.current) el.scrollTop = el.scrollHeight
  }, [campaignId, loading, messages, liveText, liveRolls.length, stage, phase])

  // A sent action should always bring the view down.
  const lastMessage = messages[messages.length - 1]
  useLayoutEffect(() => {
    if (lastMessage?.role === 'player' && lastMessage.pending && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
      nearBottomRef.current = true
    }
  }, [lastMessage])

  return (
    <div
      ref={scrollRef}
      onScroll={onScroll}
      className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden overscroll-contain px-3 py-6 sm:px-6"
    >
      <div
        className="mx-auto flex max-w-3xl flex-col gap-5"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-busy={streaming || loading || undefined}
        aria-label="Хроніка розслідування"
      >
        {loading && messages.length === 0 && (
          <div className="flex items-center justify-center gap-2 py-16 text-ash-500" role="status">
            <Loader2 size={16} className="animate-spin" aria-hidden />
            Відкриваємо справу…
          </div>
        )}

        {loadError && messages.length === 0 && (
          <div className="mx-auto max-w-md py-12 text-center text-ash-400">
            <p>Не вдалося завантажити історію кампанії.</p>
            <button
              type="button"
              onClick={onReload}
              className="mt-3 rounded border border-ash-700 px-3 py-1.5 text-sm text-parchment hover:border-ember/60 hover:bg-ember/10"
            >
              Спробувати ще раз
            </button>
          </div>
        )}

        {messages.map((message) =>
          message.role === 'keeper' ? (
            <div key={message.id} className="flex flex-col gap-3">
              <RollList rolls={message.rolls} animate={!message.fromStream} />
              <MessageBubble message={message} animate={!message.fromStream} />
            </div>
          ) : (
            <MessageBubble key={message.id} message={message} />
          ),
        )}

        {showLive && (
          <div className="flex flex-col gap-3">
            <RollList rolls={liveRolls} />
            {liveText ? (
              <MessageBubble
                message={{ role: 'keeper', text: liveText }}
                streaming={streaming}
                interrupted={failed}
              />
            ) : (
              streaming && (
                <div className="flex items-start gap-2">
                  <KeeperAvatar />
                  <div className="rounded-xl rounded-tl-sm border border-ash-700/60 bg-ash-900 px-4 py-3">
                    <span className="inline-flex gap-1" aria-hidden>
                      <span className="animate-flicker text-ash-400">.</span>
                      <span className="animate-flicker text-ash-400 [animation-delay:300ms]">.</span>
                      <span className="animate-flicker text-ash-400 [animation-delay:600ms]">.</span>
                    </span>
                  </div>
                </div>
              )
            )}
            {streaming && <StageIndicator stage={liveText ? 'narrate' : stage} />}
          </div>
        )}
      </div>
    </div>
  )
}
