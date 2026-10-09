interface Props {
  label: string
  value: string
  note?: string
  tone?: 'default' | 'alert'
}

export default function StatCard({ label, value, note, tone = 'default' }: Props) {
  const color =
    tone === 'alert' ? 'text-rose-600 dark:text-rose-400' : 'text-slate-900 dark:text-white'
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <div className={`mt-1 font-mono text-2xl font-semibold tabular-nums ${color}`}>{value}</div>
      {note && <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{note}</div>}
    </div>
  )
}
