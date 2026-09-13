import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import { Eye, Search } from 'lucide-react'

const MARKDOWN_COMPONENTS = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer nofollow" />,
}

const DISALLOWED = ['img']

export const KeeperMarkdown = memo(function KeeperMarkdown({ text }) {
  return (
    <ReactMarkdown disallowedElements={DISALLOWED} unwrapDisallowed components={MARKDOWN_COMPONENTS}>
      {text}
    </ReactMarkdown>
  )
})

export function KeeperAvatar() {
  return (
    <div
      className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-ember/40 bg-ember/15"
      aria-hidden
    >
      <Eye size={14} className="text-ember" />
    </div>
  )
}

/**
 * @param {{message: {role: string, text: string}, streaming?: boolean, interrupted?: boolean}} props
 */
function MessageBubble({ message, streaming = false, interrupted = false, animate = true }) {
  const fade = animate ? 'animate-fade-in' : ''
  const text = typeof message?.text === 'string' ? message.text : ''

  if (message?.role === 'player') {
    return (
      <div className={`flex justify-end ${fade}`}>
        <div className="flex max-w-[92%] items-start gap-2 sm:max-w-[78%]">
          <div className="sr-only">Слідчий:</div>
          <div
            className={`min-w-0 whitespace-pre-wrap break-words rounded-xl rounded-tr-sm border border-ash-700 bg-ash-800 px-4 py-3 leading-relaxed text-parchment
              ${message.pending ? 'opacity-80' : ''}`}
          >
            {text}
          </div>
          <div
            className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ash-700"
            aria-hidden
          >
            <Search size={13} className="text-ash-300" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`flex justify-start ${fade}`}>
      <div className="flex min-w-0 max-w-full items-start gap-2 sm:max-w-[88%]">
        <KeeperAvatar />
        <div className="sr-only">Хранитель:</div>
        <div
          className={`prose-keeper min-w-0 rounded-xl rounded-tl-sm border px-4 py-3
            ${interrupted ? 'border-ash-700/40 bg-ash-900/50 text-ash-400' : 'border-ash-700/60 bg-ash-900'}
            ${streaming ? 'stream-caret' : ''}`}
          aria-busy={streaming || undefined}
        >
          <KeeperMarkdown text={text} />
          {interrupted && <p className="!mt-2 text-xs italic text-ash-500">(розповідь перервано)</p>}
        </div>
      </div>
    </div>
  )
}

export default memo(MessageBubble)
