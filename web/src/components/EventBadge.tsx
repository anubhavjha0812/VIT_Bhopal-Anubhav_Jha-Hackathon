const CREDIT = ['Default', 'Credit downgrade', 'Covenant breach']

/** Impact as a place on the 1–10 scale: the number, then ten ticks filled up to it. Ticks above
 *  the credit trigger are brick, 5 and up amber, the rest navy, so a row reads like a rating ladder. */
export function ImpactPill({ impact, trigger = 7 }: { impact: number; trigger?: number }) {
  const tone = (step: number) =>
    step > impact
      ? 'bg-slate-200 dark:bg-slate-700'
      : step > trigger
        ? 'bg-rose-600 dark:bg-rose-500'
        : step >= 5
          ? 'bg-amber-500'
          : 'bg-slate-500 dark:bg-slate-400'
  return (
    <span title={`impact ${impact} of 10`} className="inline-flex items-center gap-1.5">
      <span
        className={`min-w-4 text-right font-figures text-base leading-none font-semibold tabular-nums ${
          impact > trigger ? 'text-rose-600 dark:text-rose-400' : ''
        }`}
      >
        {impact}
      </span>
      <span className="flex h-3 items-end gap-px" aria-hidden>
        {Array.from({ length: 10 }, (_, i) => (
          <span key={i} className={`w-1 rounded-[1px] ${tone(i + 1)}`} style={{ height: `${40 + i * 6.5}%` }} />
        ))}
      </span>
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
