const CAP = 0.15 // Module A's per-stock cap: a full bar

/** One bar per stock, largest first; the tick marks equal weight. */
export default function WeightBars({ weights }: { weights: Record<string, number | string> }) {
  const entries = Object.entries(weights)
    .filter(([k]) => k !== 'date')
    .map(([k, v]) => [k, Number(v)] as const)
    .sort((a, b) => b[1] - a[1])
  const equal = 1 / entries.length
  return (
    <ul className="space-y-1.5">
      {entries.map(([t, w]) => (
        <li key={t} className="flex items-center gap-2 text-sm">
          <span className="w-14 font-mono">{t}</span>
          <div className="relative h-3 flex-1 rounded bg-slate-100 dark:bg-slate-800">
            <div
              className="absolute h-3 rounded bg-teal-500 transition-[width] duration-500"
              style={{ width: `${(w / CAP) * 100}%` }}
            />
            <div className="absolute top-[-2px] h-4 w-px bg-slate-500" style={{ left: `${(equal / CAP) * 100}%` }} />
          </div>
          <span className="w-12 text-right font-mono tabular-nums">{(w * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  )
}
