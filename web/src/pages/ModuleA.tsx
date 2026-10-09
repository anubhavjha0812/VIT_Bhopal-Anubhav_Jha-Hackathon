import { useApi } from '../api'
import Chart from '../components/Chart'
import ErrorBox from '../components/ErrorBox'
import { chartColors, useTheme } from '../theme'
import WeightBars from '../components/WeightBars'

type Metrics = Record<string, Record<string, number>>
type WeightRow = { date: string } & Record<string, number>
type EquityRow = { date: string } & Record<string, number>

const NAMES: Record<string, string> = { equal: 'Equal weight (benchmark)', sentiment: 'Sentiment tilt' }
const METRIC_ROWS: [string, string, (x: number) => string][] = [
  ['total_return', 'Total return', (x) => `${(x * 100).toFixed(1)}%`],
  ['ann_vol', 'Volatility (annualised)', (x) => `${(x * 100).toFixed(1)}%`],
  ['sharpe', 'Sharpe ratio', (x) => x.toFixed(2)],
  ['max_drawdown', 'Max drawdown', (x) => `${(x * 100).toFixed(1)}%`],
  ['avg_daily_turnover', 'Avg daily turnover', (x) => `${(x * 100).toFixed(1)}%`],
]
// 15 distinct colours, stable per ticker
const PALETTE = [
  '#0d9488', '#2563eb', '#d97706', '#dc2626', '#7c3aed', '#059669', '#db2777', '#0891b2',
  '#65a30d', '#ea580c', '#4f46e5', '#ca8a04', '#be123c', '#0284c7', '#9333ea',
]

function EquityChart({ rows }: { rows: EquityRow[] }) {
  const c = chartColors(useTheme())
  const keys = Object.keys(rows[0]).filter((k) => k !== 'date')
  const option = {
    backgroundColor: 'transparent',
    legend: { top: 0, textStyle: { color: c.text } },
    grid: { left: 50, right: 20, top: 36, bottom: 30 },
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => `$${v.toFixed(3)}` },
    xAxis: { type: 'time', axisLabel: { color: c.text }, axisLine: { lineStyle: { color: c.grid } } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: c.text }, splitLine: { lineStyle: { color: c.grid } } },
    series: keys.map((k) => ({
      name: NAMES[k] ?? k,
      type: 'line',
      showSymbol: false,
      lineStyle: { width: 2, type: k === 'equal' ? 'dashed' : 'solid' },
      itemStyle: { color: k === 'equal' ? c.text : c.up },
      data: rows.map((r) => [r.date, r[k]]),
    })),
  }
  return <Chart option={option} style={{ height: 300 }} />
}

function WeightsChart({ rows }: { rows: WeightRow[] }) {
  const c = chartColors(useTheme())
  const tickers = Object.keys(rows[0]).filter((k) => k !== 'date')
  const option = {
    backgroundColor: 'transparent',
    legend: { top: 0, type: 'scroll', textStyle: { color: c.text } },
    grid: { left: 50, right: 20, top: 40, bottom: 30 },
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${(v * 100).toFixed(1)}%` },
    xAxis: { type: 'time', axisLabel: { color: c.text }, axisLine: { lineStyle: { color: c.grid } } },
    yAxis: {
      type: 'value',
      max: 1,
      axisLabel: { color: c.text, formatter: (v: number) => `${v * 100}%` },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: tickers.map((t, i) => ({
      name: t,
      type: 'line',
      stack: 'w',
      showSymbol: false,
      lineStyle: { width: 0 },
      areaStyle: { opacity: 0.85 },
      itemStyle: { color: PALETTE[i % PALETTE.length] },
      data: rows.map((r) => [r.date, r[t]]),
    })),
  }
  return <Chart option={option} style={{ height: 360 }} />
}

const card = 'rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900'
const head = 'border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800'

export default function ModuleA() {
  const metrics = useApi<Metrics>('/module-a/metrics')
  const weights = useApi<WeightRow[]>('/module-a/weights')
  const equity = useApi<EquityRow[]>('/module-a/equity')
  const error = metrics.error ?? weights.error ?? equity.error
  if (error) return <ErrorBox message={error} />
  if (!metrics.data || !weights.data || !equity.data) return <p className="text-sm text-slate-500">Loading…</p>

  const strategies = Object.keys(metrics.data)
  const latest = weights.data[weights.data.length - 1]
  return (
    <div className="space-y-6">
      <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
        A mock index of 15 S&amp;P 100 stocks. Each day the sentiment tilt moves weight toward stocks with better recent
        news (3-day mean sentiment), long only, capped at 15% per stock, and trades at the <em>next</em> close so no
        future news is used. 5 bp cost on every trade. Equal weight is the benchmark.
      </p>

      <div className="grid gap-6 xl:grid-cols-3">
        <section className={`${card} xl:col-span-2`}>
          <h2 className={head}>Growth of $1, after costs</h2>
          <div className="p-2">
            <EquityChart rows={equity.data} />
          </div>
        </section>
        <section className={card}>
          <h2 className={head}>Performance, {equity.data[0].date} → {equity.data[equity.data.length - 1].date}</h2>
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
              <tr>
                <th className="px-4 py-2" />
                {strategies.map((s) => (
                  <th key={s} className="px-4 py-2 text-right">
                    {s === 'equal' ? 'Equal' : 'Tilt'}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map(([key, label, fmt]) => (
                <tr key={key} className="border-t border-slate-100 dark:border-slate-800">
                  <td className="px-4 py-2">{label}</td>
                  {strategies.map((s) => (
                    <td key={s} className="px-4 py-2 text-right font-mono tabular-nums">
                      {fmt(metrics.data![s][key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400">
            Honest result: over these two months the tilt trails equal weight. 43 days is too short to tell skill from
            luck.
          </p>
        </section>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <section className={`${card} xl:col-span-2`}>
          <h2 className={head}>Weights over time (sentiment tilt)</h2>
          <div className="p-2">
            <WeightsChart rows={weights.data} />
          </div>
        </section>
        <section className={card}>
          <h2 className={head}>Weights on {latest.date}</h2>
          <div className="p-4">
            <WeightBars weights={latest} />
            <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
              Bar full = 15% cap. Tick = equal weight ({(100 / 15).toFixed(1)}%).
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
