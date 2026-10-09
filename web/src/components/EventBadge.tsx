const CREDIT = ['Default', 'Credit downgrade', 'Covenant breach']

export function ImpactPill({ impact, trigger = 7 }: { impact: number; trigger?: number }) {
  const hot = impact > trigger
  return (
    <span
      title={`impact ${impact} of 10`}
      className={`inline-block min-w-7 rounded px-1.5 py-0.5 text-center font-mono text-xs font-semibold ${
        hot
          ? 'bg-rose-600 text-white'
          : impact >= 5
            ? 'bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200'
            : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
      }`}
    >
      {impact}
    </span>
  )
}

export default function EventBadge({ event }: { event: string }) {
  const credit = CREDIT.includes(event)
  const other = event === 'Other'
  return (
    <span
      className={`inline-block whitespace-nowrap rounded px-1.5 py-0.5 text-xs font-medium ${
        credit
          ? 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300'
          : other
            ? 'text-slate-500 dark:text-slate-400'
            : 'bg-teal-100 text-teal-800 dark:bg-teal-950/60 dark:text-teal-300'
      }`}
    >
      {event}
    </span>
  )
}
