import { Sword, Plus, ScrollText } from 'lucide-react'

const CAMPAIGNS = [
  { id: 1, name: 'The Ashen Crown', active: true },
  { id: 2, name: 'Sunken City', active: false },
]

export default function Sidebar({ onNewGame }) {
  return (
    <aside className="w-56 shrink-0 flex flex-col border-r border-ash-800 bg-ash-900/60">
      {/* Logo */}
      <div className="px-4 py-5 border-b border-ash-800">
        <div className="flex items-center gap-2">
          <Sword size={18} className="text-ember" />
          <span className="font-display text-sm tracking-widest text-parchment uppercase">
            GM Assistant
          </span>
        </div>
      </div>

      {/* Campaigns */}
      <div className="flex-1 overflow-y-auto px-3 py-4">
        <p className="text-xs tracking-widest text-ash-500 uppercase mb-3 px-1">
          Campaigns
        </p>
        <ul className="space-y-1">
          {CAMPAIGNS.map((c) => (
            <li key={c.id}>
              <button
                className={`w-full text-left px-3 py-2 rounded text-sm font-body transition-colors flex items-center gap-2
                  ${c.active
                    ? 'bg-ash-800 text-parchment'
                    : 'text-ash-400 hover:text-ash-200 hover:bg-ash-800/50'
                  }`}
              >
                <ScrollText size={13} className="shrink-0 text-ember/70" />
                {c.name}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* New game */}
      <div className="px-3 py-4 border-t border-ash-800">
        <button
          onClick={onNewGame}
          className="w-full flex items-center gap-2 px-3 py-2 rounded text-sm text-ash-400
            hover:text-parchment hover:bg-ash-800/60 transition-colors"
        >
          <Plus size={14} />
          New Campaign
        </button>
      </div>
    </aside>
  )
}
