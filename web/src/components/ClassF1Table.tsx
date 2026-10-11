import type { ScoreRow } from '../types'

interface Props {
  rows: ScoreRow[]
  engineModel: string
}

const pct = (x: number) => (x * 100).toFixed(1)

/** F1 per event class for each model, with the engine model's 95% interval under its score. */
export default function ClassF1Table({ rows, engineModel }: Props) {
  const scored = rows.filter((r) => r.per_class_f1)
  if (!scored.length) return null
  const classes = Object.keys(scored[0].per_class_f1!)
  return (
    <section className="min-w-0 rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <h2 className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">
        F1 per event class <span className="font-normal text-slate-500">· %, same test</span>
      </h2>
      <div className="overflow-x-auto">
        <table className="min-w-[780px] w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="px-3 py-2">Class</th>
              {scored.map((r) => (
                <th
                  key={r.model}
                  className={`px-3 py-2 text-right font-medium normal-case tracking-normal ${
                    r.model === engineModel ? 'bg-teal-50 font-semibold text-slate-800 dark:bg-teal-950/40 dark:text-slate-100' : ''
                  }`}
                >
                  {r.model}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {classes.map((c) => (
              <tr key={c} className="border-t border-slate-100 dark:border-slate-800">
                <td className="px-3 py-2 whitespace-nowrap">{c}</td>
                {scored.map((r) => {
                  const engine = r.model === engineModel
                  const ci = r.per_class_f1_ci95?.[c]
                  return (
                    <td
                      key={r.model}
                      className={`px-3 py-2 text-right font-mono tabular-nums ${engine ? 'bg-teal-50 font-semibold dark:bg-teal-950/40' : ''}`}
                    >
                      {pct(r.per_class_f1![c])}
                      {engine && ci && (
                        <div className="text-[10px] font-normal text-slate-400">
                          {pct(ci[0])}–{pct(ci[1])}
                        </div>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
