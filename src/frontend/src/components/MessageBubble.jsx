import ReactMarkdown from 'react-markdown'
import { Sword, User } from 'lucide-react'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="flex items-start gap-2 max-w-[75%]">
          <div className="bg-ash-800 border border-ash-700 rounded-xl rounded-tr-sm px-4 py-3 text-base text-parchment leading-relaxed">
            {message.content}
          </div>
          <div className="shrink-0 w-7 h-7 rounded-full bg-ash-700 flex items-center justify-center mt-0.5">
            <User size={13} className="text-ash-300" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex items-start gap-2 max-w-[85%]">
        <div className="shrink-0 w-7 h-7 rounded-full bg-ember/20 border border-ember/40 flex items-center justify-center mt-0.5">
          <Sword size={12} className="text-ember" />
        </div>
        <div className="bg-ash-900 border border-ash-700/60 rounded-xl rounded-tl-sm px-4 py-3 text-base leading-relaxed prose-gm">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
