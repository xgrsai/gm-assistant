import { Sword } from 'lucide-react'

export default function TypingIndicator() {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex items-start gap-2">
        <div className="shrink-0 w-7 h-7 rounded-full bg-ember/20 border border-ember/40 flex items-center justify-center mt-0.5">
          <Sword size={12} className="text-ember" />
        </div>
        <div className="bg-ash-900 border border-ash-700/60 rounded-xl rounded-tl-sm px-4 py-3">
          <span className="text-sm text-ash-400 italic tracking-wide">
            Game Master is thinking
            <span className="inline-flex gap-0.5 ml-1">
              <span className="animate-blink" style={{ animationDelay: '0ms' }}>.</span>
              <span className="animate-blink" style={{ animationDelay: '200ms' }}>.</span>
              <span className="animate-blink" style={{ animationDelay: '400ms' }}>.</span>
            </span>
          </span>
        </div>
      </div>
    </div>
  )
}
