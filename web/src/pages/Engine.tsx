import { useMemo, useState } from 'react'
import { signed } from '../format'
import { useSearchParams } from 'react-router-dom'
import { useApi } from '../api'
import Chart from '../components/Chart'
import ErrorBox from '../components/ErrorBox'
import EventBadge, { ImpactPill } from '../components/EventBadge'
import StatCard from '../components/StatCard'
import { AnalystTag } from '../components/Tickers'
import { chartColors, useTheme } from '../theme'
import type { DailyRow, PricePoint, Signal } from '../types'

const MIN_IMPACT_CHOICES = [1, 5, 6, 7, 8]
const MAX_ROWS = 200

function TickerPicker({ tickers, value, onChange }: { tickers: string[]; value: string; onChange: (t: string) => void }) {
  return (
    <div className="no-scrollbar -mx-3 flex gap-2 overflow-x-auto px-3 pb-1 sm:mx-0 sm:flex-wrap sm:px-0 sm:pb-0">
      {tickers.map((t) => (
        <button
          key={t}
          type="button"
          onClick={() => onChange(t)}
          className={`shrink-0 rounded-lg border px-3 py-1.5 font-figures text-sm transition ${
            t === value
              ? 'border-teal-600 bg-teal-600 text-white'
              : 'border-slate-200 bg-white text-slate-700 hover:border-teal-400 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200'
          }`}
        >
          {t}
        </button>
      ))}
    </div>
  )
}

function SentimentPriceChart({ daily, prices }: { daily: DailyRow[]; prices: PricePoint[] }) {
  const c = chartColors(useTheme())
  const option = {
    backgroundColor: 'transparent',
    legend: { top: 0, textStyle: { color: c.text } },
    grid: { left: 50, right: 60, top: 36, bottom: 30 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'time', axisLabel: { color: c.text }, axisLine: { lineStyle: { color: c.grid } } },
    yAxis: [
      {
        type: 'value',
        name: 'sentiment',
        min: -1,
        max: 1,
        nameTextStyle: { color: c.text },
        axisLabel: { color: c.text },
        splitLine: { lineStyle: { color: c.grid } },
      },
      {
        type: 'value',
        name: 'close $',
        scale: true,
        nameTextStyle: { color: c.text },
        axisLabel: { color: c.text },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: 'Mean sentiment of the day',
        type: 'bar',
        itemStyle: { color: c.up }, // legend swatch; each bar sets its own colour below
        data: daily.map((d) => ({
          value: [d.date, d.sentiment],
          itemStyle: { color: d.sentiment >= 0 ? c.up : c.down },
        })),
      },
      {
        name: 'Close price',
        type: 'line',
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { width: 2, color: c.line },
        itemStyle: { color: c.line },
        data: prices.map((p) => [p.date, p.close]),
      },
    ],
  }
  return <Chart option={option} style={{ height: 320 }} />
}

function SignalTable({ rows }: { rows: Signal[] }) {
  return (
    <div className="overflow-x-auto">
        <table className="min-w-[900px] w-full text-sm">
        <thead className="text-left text-xs text-slate-500 dark:text-slate-400">
          <tr>
            <th className="px-4 py-2">Time (UTC)</th>
            <th className="px-4 py-2">Source</th>
            <th className="px-4 py-2">Event</th>
            <th className="px-4 py-2 text-right">Impact</th>
            <th className="px-4 py-2 text-right">Sentiment</th>
            <th className="px-4 py-2">Text</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.doc_id} className="border-t border-slate-100 align-top dark:border-slate-800">
              <td className="whitespace-nowrap px-4 py-2 font-figures text-xs text-slate-500">
                {s.published_at.slice(0, 16).replace('T', ' ')}
              </td>
              <td className="px-4 py-2 text-xs">{s.source}</td>
              <td className="px-4 py-2">
                <EventBadge event={s.event} />
                {s.analyst_rating && <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">analyst rating</div>}
                {s.fired && (
                  <div className="mt-1 text-xs font-medium text-rose-700 dark:text-rose-400" title={s.reason}>
                    → Module B
                  </div>
                )}
              </td>
              <td className="px-4 py-2 text-right">
                <ImpactPill impact={s.impact} />
              </td>
              <td
                className={`px-4 py-2 text-right font-figures tabular-nums ${
                  s.sentiment > 0.2 ? 'text-teal-700 dark:text-teal-400' : s.sentiment < -0.2 ? 'text-rose-600 dark:text-rose-400' : ''
                }`}
              >
                {signed(s.sentiment)}
              </td>
              <td className="min-w-72 px-4 py-2">
                {s.role === 'analyst' && (
                  <span className="mr-2">
                    <AnalystTag ticker={s.ticker} />
                  </span>
                )}
                {s.text}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Engine() {
  const [params, setParams] = useSearchParams()
  const tickers = useApi<string[]>('/tickers')
  const ticker = params.get('t') ?? tickers.data?.[0] ?? ''
  const [minImpact, setMinImpact] = useState(6)

  if (tickers.error) return <ErrorBox message={tickers.error} />
  if (!tickers.data) return <p className="text-sm text-slate-500">Loading…</p>

  return (
    <div className="space-y-6">
      <TickerPicker tickers={tickers.data} value={ticker} onChange={(t) => setParams({ t })} />
      {/* key: a new ticker remounts the panel, so its data loads fresh */}
      <TickerPanel key={ticker} ticker={ticker} minImpact={minImpact} setMinImpact={setMinImpact} />
    </div>
  )
}

function TickerPanel({ ticker, minImpact, setMinImpact }: { ticker: string; minImpact: number; setMinImpact: (n: number) => void }) {
  const daily = useApi<DailyRow[]>(`/daily?ticker=${ticker}`)
  const prices = useApi<PricePoint[]>(`/prices?ticker=${ticker}`)
  const signals = useApi<Signal[]>(`/signals?ticker=${ticker}&limit=5000`)

  const stats = useMemo(() => {
    if (!signals.data || !prices.data?.length) return null
    const s = signals.data.filter((r) => r.role !== 'analyst')
    const first = prices.data[0].close
    const last = prices.data[prices.data.length - 1].close
    return {
      texts: s.length,
      mean: s.reduce((a, r) => a + r.sentiment, 0) / s.length,
      high: s.filter((r) => r.impact > 7).length,
      move: last / first - 1,
    }
  }, [signals.data, prices.data])

  const shown = useMemo(
    () =>
      (signals.data ?? [])
        .filter((s) => s.impact >= minImpact)
        .sort((a, b) => b.impact - a.impact || b.published_at.localeCompare(a.published_at))
        .slice(0, MAX_ROWS),
    [signals.data, minImpact],
  )

  const error = daily.error ?? prices.error ?? signals.error
  if (error) return <ErrorBox message={error} />
  if (!daily.data || !prices.data || !signals.data || !stats) {
    return <p className="text-sm text-slate-500">Loading {ticker}…</p>
  }
  const matching = signals.data.filter((s) => s.impact >= minImpact).length

  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2 sm:gap-4 lg:grid-cols-4">
        <StatCard hero label={`${ticker} texts`} value={stats.texts.toLocaleString()} note="news + tweets naming the stock" />
        <StatCard
          hero
          label="Mean sentiment"
          value={signed(stats.mean, 3)}
          note="−1 very negative · +1 very positive"
        />
        <StatCard hero label="High impact (> 7)" value={String(stats.high)} tone="alert" note="texts above the trigger" />
        <StatCard
          hero
          label="Price over window"
          value={`${signed(stats.move * 100, 1)}%`}
          note="first to last close"
        />
      </div>

      <section className="rounded-xl border border-slate-200 bg-white p-2 dark:border-slate-800 dark:bg-slate-900">
        <SentimentPriceChart daily={daily.data} prices={prices.data} />
        <p className="px-2 pb-1 text-xs text-slate-500 dark:text-slate-400">
          Bars: mean sentiment of the texts published that day (UTC). Line: daily close. Descriptive only; the
          backtest in Module A trades a day later to avoid look-ahead.
        </p>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="flex flex-col gap-3 border-b border-slate-200 px-4 py-3 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
          <h2 className="text-lg font-semibold">
            Texts with impact ≥ {minImpact}{' '}
            <span className="font-normal text-slate-500">
              · {matching.toLocaleString()}
              {matching > MAX_ROWS ? `, top ${MAX_ROWS} shown` : ''}
            </span>
          </h2>
          <label className="flex items-center gap-2 text-xs text-slate-500">
            Min impact
            <select
              value={minImpact}
              onChange={(e) => setMinImpact(Number(e.target.value))}
              className="rounded border border-slate-200 bg-white px-2 py-1 text-slate-800 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
            >
              {MIN_IMPACT_CHOICES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
        <SignalTable rows={shown} />
      </section>
    </>
  )
}
