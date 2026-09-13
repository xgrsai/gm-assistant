/** Compact marker for campaigns whose Keeper uses the knowledge base (`use_rag: true`). */
export default function RagBadge({ className = '' }) {
  return (
    <span
      title="Хранитель спирається на базу правил і лору (RAG)"
      className={`inline-flex shrink-0 items-center rounded-sm border border-verdigris/50 bg-verdigris/10 px-1 font-mono text-[9px] font-semibold uppercase leading-[1.35] tracking-wider text-verdigris ${className}`}
    >
      RAG
      <span className="sr-only"> — з базою правил</span>
    </span>
  )
}
