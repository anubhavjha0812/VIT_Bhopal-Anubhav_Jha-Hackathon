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
    <div className="min-w-0 rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900 sm:p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <div className={`mt-1 break-words font-mono text-xl font-semibold tabular-nums sm:text-2xl ${color}`}>{value}</div>
      {note && <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{note}</div>}
    </div>
  )
}
