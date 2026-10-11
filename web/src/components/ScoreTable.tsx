import type { ScoreRow } from '../types'

interface Props {
  title: string
  rows: ScoreRow[]
  engineModel: string
}

const pct = (x: number) => (x * 100).toFixed(1)
const ci = ([lo, hi]: [number, number]) => `${pct(lo)}–${pct(hi)}`

export default function ScoreTable({ title, rows, engineModel }: Props) {
  const sorted = [...rows].sort((a, b) => b.accuracy - a.accuracy)
  return (
    <section className="min-w-0 rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <h2 className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">
        {title} <span className="font-normal text-slate-500">· n = {rows[0]?.n.toLocaleString()}</span>
      </h2>
      <div className="overflow-x-auto">
        <table className="min-w-[640px] w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="px-3 py-2">Model</th>
              <th className="px-3 py-2 text-right">Accuracy %</th>
              <th className="px-3 py-2 text-right">Macro-F1 %</th>
              <th className="px-3 py-2 text-right">Texts/s</th>
              <th className="px-3 py-2 text-right">Size MB</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const engine = r.model === engineModel
              const baseline = r.model.includes('baseline')
              return (
                <tr
                  key={r.model}
                  className={`border-t border-slate-100 dark:border-slate-800 ${
                    engine ? 'bg-teal-50 dark:bg-teal-950/40' : ''
                  }`}
                >
                  <td className="max-w-72 px-3 py-2">
                    <span className={engine ? 'font-semibold' : ''}>{r.model}</span>
                    {engine && (
                      <span className="ml-2 rounded bg-teal-600 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                        ENGINE
                      </span>
                    )}
                    {baseline && (
                      <span className="ml-2 rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700 dark:bg-slate-700 dark:text-slate-200">
                        BASELINE
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">
                    {pct(r.accuracy)}
                    <div className="text-[10px] text-slate-400">{ci(r.accuracy_ci95)}</div>
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">
                    {pct(r.macro_f1)}
                    <div className="text-[10px] text-slate-400">{ci(r.macro_f1_ci95)}</div>
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">
                    {r.texts_per_s >= 1000 ? Math.round(r.texts_per_s).toLocaleString() : r.texts_per_s.toFixed(0)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{r.model_size_mb.toFixed(0)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="px-4 py-2 text-xs text-slate-500 dark:text-slate-400">
        Public held-out test set, scored once per model. Small numbers: 95% bootstrap interval. Speed on CPU, batch 64.
      </p>
    </section>
  )
}
