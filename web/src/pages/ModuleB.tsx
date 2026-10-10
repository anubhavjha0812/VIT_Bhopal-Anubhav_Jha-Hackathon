import { useSearchParams } from 'react-router-dom'
import { useApi } from '../api'
import Chart from '../components/Chart'
import ErrorBox from '../components/ErrorBox'
import StatCard from '../components/StatCard'
import { chartColors, useTheme } from '../theme'

// Shapes of /module-b, /module-b/alerts and /module-b/checks (src/riskengine/api.py).
interface Capital {
  cet1_before: number
  cet1_after: number
  floor: number
  loss_usd: number
  rwa_before_inr: number
  rwa_after_inr: number
  fx_before: number
  fx_after: number
}
interface Alert extends Capital {
  doc_id: string
  published_at: string
  source: string
  ticker: string
  text: string
  event: string
  impact: number
  gate: string
  notches: number
  matched: string[]
}
interface Dist {
  expected: number
  var: number
  es: number
}
interface Report {
  book_date: string
  scenario: {
    ticker: string | null
    notches: number
    episode: string
    episode_label: string
    start: string
    end: string
    multiple: number
    rating_before: string | null
    rating_after: string | null
    d_baa_bp: number
    d_aaa_bp: number
    fx_move: number
    z: number
  }
  capital: Capital
  by_instrument: Record<string, number>
  top_losses: { position: string; ticker: string; instrument: string; rating: string; notional: number; loss: number }[]
  obligors: {
    ticker: string
    rating_before: string
    rating_after: string
    spread_before: number
    spread_after: number
    pd_before: number
    pd_after: number
    loss: number
  }[]
  stages: Record<string, number>
  monte_carlo: {
    draws: number
    level: number
    histogram: { edges: number[]; before: number[]; after: number[] }
    before: Dist
    after: Dist
  }
  reverse_stress: number | null
  memo: string
}
interface Checks {
  alerts: { ticker: string; window: string; predicted_loss: number; naive_loss: number; realised_loss: number; sign_match: boolean }[]
  crises: { mae: Record<'predicted' | 'naive', { loss: number; cet1: number; sign: number }>; rows: unknown[] }
  monthly: {
    summary: {
      origins: number
      model_mae: number
      naive_mae: number
      ratio: number
      pass_ratio: number
      passed: boolean
      model_sign: number
      naive_sign: number
      dm_p: number
    }
    rows: { origin: string; realised_loss: number; model_loss: number; naive_loss: number }[]
  }
}

const EPISODES = [
  ['covid2020', 'COVID 2020'],
  ['gfc2008', 'GFC 2008'],
  ['euro2011', 'Euro 2011'],
] as const
const INSTRUMENTS: Record<string, string> = {
  revolver: 'Revolver',
  bond: 'Bond',
  cds_bought: 'CDS bought',
  cds_sold: 'CDS sold',
  irs_pay_fixed: 'IRS pay-fixed',
}
const usd = (x: number) => `${x < 0 ? '−' : ''}$${Math.abs(x / 1e6).toFixed(1)}m`
const pct = (x: number, d = 2) => `${(100 * x).toFixed(d)}%`

const card = 'rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900'
const head = 'border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800'
const muted = 'text-xs text-slate-500 dark:text-slate-400'
const pill = (on: boolean) =>
  `rounded-lg px-3 py-1.5 text-sm ${
    on
      ? 'bg-teal-600 font-semibold text-white'
      : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
  }`

/** CET1 on a 0-15% scale: the 8% floor, before and after. */
function Cet1Gauge({ c }: { c: Capital }) {
  const top = 0.15
  const at = (x: number) => `${(100 * Math.min(Math.max(x, 0), top)) / top}%`
  const breach = c.cet1_after < c.floor
  return (
    <div className="space-y-2">
      <div className="relative h-6 overflow-hidden rounded-md bg-slate-100 dark:bg-slate-800">
        <div className="absolute inset-y-0 left-0 bg-rose-200/70 dark:bg-rose-900/50" style={{ width: at(c.floor) }} />
        <div
          className={`absolute inset-y-1 left-0 rounded-sm ${breach ? 'bg-rose-600' : 'bg-teal-600'}`}
          style={{ width: at(c.cet1_after) }}
        />
        <div className="absolute inset-y-0 w-0.5 bg-slate-900 dark:bg-white" style={{ left: at(c.cet1_before) }} />
      </div>
      <div className={`flex justify-between ${muted}`}>
        <span>0%</span>
        <span>red zone: below the {pct(c.floor, 0)} floor (RBI CET1 5.5% + buffer 2.5%)</span>
        <span>15%</span>
      </div>
      <p className="text-sm">
        <span className="font-mono">{pct(c.cet1_before)}</span> (black line) →{' '}
        <span className={`font-mono font-semibold ${breach ? 'text-rose-600 dark:text-rose-400' : ''}`}>
          {pct(c.cet1_after)}
        </span>
        {breach ? ' · below the floor' : ` · ${(100 * (c.cet1_after - c.floor)).toFixed(2)} pt above the floor`}
      </p>
    </div>
  )
}

function LossHistogram({ mc }: { mc: Report['monte_carlo'] }) {
  const c = chartColors(useTheme())
  const { edges } = mc.histogram
  // Default losses come in a few big lumps (whole companies), so most bins are empty: bars, not lines.
  const labels = edges.slice(1).map((e, i) => `${Math.round((e + edges[i]) / 2 / 1e6)}`)
  // Share of draws on a log scale, computed here (decades above the floor): bars cannot start
  // from 0 on an ECharts log axis.
  const floor = Math.floor(Math.log10(1 / mc.draws / 2)) // a whole decade, so ticks are 0.001%, 0.01%, …
  const share = (n: number) => (n > 0 ? Math.log10(n / mc.draws) - floor : null)
  const asPct = (v: number) => `${+(100 * 10 ** (v + floor)).toPrecision(2)}%`
  const bin = (x: number) => Math.min(edges.findIndex((e) => e > x) - 1, labels.length - 1)
  const line = (name: string, x: number, color: string) => ({
    xAxis: Math.max(bin(x), 0),
    lineStyle: { color, type: 'dashed' },
    label: { formatter: `${name} ${usd(x)}`, color },
  })
  const option = {
    backgroundColor: 'transparent',
    legend: { top: 0, textStyle: { color: c.text } },
    animation: false,
    grid: { left: 56, right: 24, top: 44, bottom: 40 },
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v: number | null) => (v == null ? 'none' : `${asPct(v)} of draws`),
    },
    xAxis: {
      type: 'category',
      data: labels,
      name: 'default loss, $m',
      nameLocation: 'middle',
      nameGap: 26,
      nameTextStyle: { color: c.text },
      axisLabel: { color: c.text },
      axisLine: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: -floor,
      interval: 1,
      axisLabel: { color: c.text, formatter: asPct },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: [
      {
        name: 'Before',
        type: 'bar',
        barGap: '0%',
        itemStyle: { color: c.text, opacity: 0.55 },
        data: mc.histogram.before.map(share),
      },
      {
        name: 'After',
        type: 'bar',
        itemStyle: { color: c.down },
        data: mc.histogram.after.map(share),
        markLine: {
          symbol: 'none',
          data: [line('VaR 99%', mc.after.var, c.down), line('ES 99%', mc.after.es, c.line)],
        },
      },
    ],
  }
  return <Chart option={option} style={{ height: 300 }} />
}

function MonthlyChart({ rows }: { rows: Checks['monthly']['rows'] }) {
  const c = chartColors(useTheme())
  const series = [
    ['realised_loss', 'Realised', c.line, 2],
    ['model_loss', 'State model', c.down, 1.5],
    ['naive_loss', 'Naive average', c.up, 1.5],
  ] as const
  const option = {
    backgroundColor: 'transparent',
    legend: { top: 0, textStyle: { color: c.text } },
    grid: { left: 56, right: 20, top: 36, bottom: 30 },
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => `$${v.toFixed(1)}m` },
    xAxis: { type: 'time', axisLabel: { color: c.text }, axisLine: { lineStyle: { color: c.grid } } },
    yAxis: {
      type: 'value',
      name: 'book loss, $m',
      axisLabel: { color: c.text },
      nameTextStyle: { color: c.text },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: series.map(([key, name, color, width]) => ({
      name,
      type: 'line',
      showSymbol: false,
      lineStyle: { width },
      itemStyle: { color },
      data: rows.map((r) => [r.origin, r[key] / 1e6]),
    })),
  }
  return <Chart option={option} style={{ height: 280 }} />
}

function Picker({ alerts, tickers, doc }: { alerts: Alert[]; tickers: string[]; doc: string | null }) {
  const [params, setParams] = useSearchParams()
  const episode = params.get('episode') ?? 'covid2020'
  const whatIf = !doc && params.has('ticker')
  const marketOnly = params.has('market')
  const set = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(patch)) {
      if (v == null) next.delete(k)
      else next.set(k, v)
    }
    setParams(next, { replace: true })
  }
  return (
    <section className={card}>
      <h2 className={head}>Scenario</h2>
      <div className="space-y-4 p-4">
        <div>
          <div className={`mb-2 ${muted}`}>Real alerts the credit gate passed (Apr–Jun 2020)</div>
          <div className="grid gap-2 sm:grid-cols-2">
            {alerts.map((a) => (
              <button
                key={`${a.doc_id}-${a.ticker}`}
                onClick={() => set({ doc: a.doc_id, ticker: a.ticker, notches: null, market: null })}
                className={`rounded-lg border p-3 text-left text-sm ${
                  doc === a.doc_id
                    ? 'border-teal-600 bg-teal-50 dark:bg-teal-950/40'
                    : 'border-slate-200 hover:border-slate-400 dark:border-slate-700'
                }`}
              >
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-mono font-semibold">{a.ticker}</span>
                  <span className="text-slate-500">{a.published_at.slice(0, 10)} · {a.source}</span>
                  <span className="ml-auto font-mono">CET1 {pct(a.cet1_after)}</span>
                </div>
                <div className="mt-1 line-clamp-2">{a.text}</div>
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <button className={pill(marketOnly)} onClick={() => set({ doc: null, ticker: null, notches: null, market: '1' })}>
            Market only
          </button>
          <label className="text-sm">
            <span className={`block ${muted}`}>What-if: company</span>
            <select
              value={whatIf ? (params.get('ticker') ?? '') : ''}
              onChange={(e) =>
                set(
                  e.target.value
                    ? { doc: null, market: null, ticker: e.target.value, notches: params.get('notches') ?? '1' }
                    : { ticker: null, market: '1' },
                )
              }
              className="mt-1 rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-700 dark:bg-slate-900"
            >
              <option value="">—</option>
              {tickers.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className={`block ${muted}`}>notches cut</span>
            <select
              disabled={!whatIf}
              value={params.get('notches') ?? '1'}
              onChange={(e) => set({ notches: e.target.value })}
              className="mt-1 rounded-lg border border-slate-300 bg-white px-2 py-1.5 disabled:opacity-40 dark:border-slate-700 dark:bg-slate-900"
            >
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <option key={n}>{n}</option>
              ))}
              <option value="99">default</option>
            </select>
          </label>
          <div className="ml-auto">
            <span className={`block ${muted}`}>Market episode (real FRED and price moves)</span>
            <div className="mt-1 flex flex-wrap gap-2">
              {EPISODES.map(([key, label]) => (
                <button key={key} className={pill(episode === key)} onClick={() => set({ episode: key })}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function ReportView({ rep }: { rep: Report }) {
  const { scenario: s, capital: c, monte_carlo: mc } = rep
  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="CET1 ratio"
          value={`${pct(c.cet1_before, 1)} → ${pct(c.cet1_after, 1)}`}
          note={`floor ${pct(c.floor, 0)}`}
          tone={c.cet1_after < c.floor ? 'alert' : 'default'}
        />
        <StatCard label="Loss before tax" value={usd(c.loss_usd)} note={`book of ${rep.book_date}`} />
        <StatCard
          label={`ES ${pct(mc.level, 0)} (defaults)`}
          value={`${usd(mc.before.es)} → ${usd(mc.after.es)}`}
          note={`${mc.draws.toLocaleString()} correlated draws`}
        />
        <StatCard
          label="Reverse stress"
          value={rep.reverse_stress == null ? '> 20×' : `${rep.reverse_stress.toFixed(2)}×`}
          note={`of ${s.episode_label} takes CET1 to ${pct(c.floor, 0)}`}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        <section className={`${card} xl:col-span-3`}>
          <h2 className={head}>Risk memo</h2>
          <div className="space-y-4 p-4">
            <Cet1Gauge c={c} />
            <div className="space-y-3 text-sm leading-relaxed whitespace-pre-line">{rep.memo}</div>
            <p className={muted}>Fixed template, no LLM: every number above comes from this run.</p>
          </div>
        </section>
        <section className={`${card} min-w-0 xl:col-span-2`}>
          <h2 className={head}>Top 10 position losses</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                <tr>
                  <th className="px-4 py-2">Name</th>
                  <th className="px-4 py-2">Instrument</th>
                  <th className="px-4 py-2 text-right">Notional</th>
                  <th className="px-4 py-2 text-right">Loss</th>
                </tr>
              </thead>
              <tbody>
                {rep.top_losses.map((p) => (
                  <tr key={p.position} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="px-4 py-2 font-mono">
                      {p.ticker} <span className="text-xs text-slate-500">{p.rating}</span>
                    </td>
                    <td className="px-4 py-2">{INSTRUMENTS[p.instrument] ?? p.instrument}</td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums">{usd(p.notional)}</td>
                    <td className={`px-4 py-2 text-right font-mono tabular-nums ${p.loss < 0 ? 'text-teal-600' : ''}`}>
                      {usd(p.loss)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className={`flex flex-wrap gap-x-4 gap-y-1 border-t border-slate-200 px-4 py-3 dark:border-slate-800 ${muted}`}>
            {Object.entries(rep.by_instrument).map(([k, v]) => (
              <span key={k}>
                {INSTRUMENTS[k] ?? k}: <span className="font-mono">{usd(v)}</span>
              </span>
            ))}
          </div>
        </section>
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        <section className={`${card} xl:col-span-3`}>
          <h2 className={head}>One-year default loss: before vs after (Gaussian copula)</h2>
          <div className="p-2">
            <LossHistogram mc={mc} />
          </div>
          <p className={`px-4 pb-3 ${muted}`}>
            Before: long-run PDs, calm 2008–2019 correlation. After: scenario PDs, correlation of 14 Feb – 23 Mar 2020.
            VaR {usd(mc.before.var)} → {usd(mc.after.var)}.
          </p>
        </section>
        <section className={`${card} min-w-0 xl:col-span-2`}>
          <h2 className={head}>Per company</h2>
          <div className="max-h-96 overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white text-left text-xs uppercase tracking-wide text-slate-500 dark:bg-slate-900 dark:text-slate-400">
                <tr>
                  <th className="px-3 py-2">Name</th>
                  <th className="px-3 py-2">Rating</th>
                  <th className="min-w-28 px-3 py-2 text-right whitespace-nowrap">Spread bp</th>
                  <th className="px-3 py-2 text-right">PD</th>
                  <th className="px-3 py-2 text-right">Loss</th>
                </tr>
              </thead>
              <tbody>
                {[...rep.obligors]
                  .sort((a, b) => b.loss - a.loss)
                  .map((o) => (
                    <tr key={o.ticker} className="border-t border-slate-100 dark:border-slate-800">
                      <td className="px-3 py-1.5 font-mono">{o.ticker}</td>
                      <td className="px-3 py-1.5 font-mono text-xs whitespace-nowrap">
                        {o.rating_before === o.rating_after ? o.rating_before : `${o.rating_before} → ${o.rating_after}`}
                      </td>
                      <td className="min-w-28 px-3 py-1.5 text-right font-mono tabular-nums whitespace-nowrap">
                        {Math.round(100 * o.spread_before)} → {Math.round(100 * o.spread_after)}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono tabular-nums whitespace-nowrap">
                        {pct(o.pd_before)} → {pct(o.pd_after)}
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono tabular-nums">{usd(o.loss)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </>
  )
}

const box = 'min-w-0 rounded-lg border border-slate-200 p-4 dark:border-slate-800'

const TONES = {
  good: 'bg-teal-100 text-teal-800 dark:bg-teal-900/50 dark:text-teal-300',
  bad: 'bg-rose-100 text-rose-800 dark:bg-rose-900/50 dark:text-rose-300',
  neutral: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
}

function Verdict({ tone, text }: { tone: keyof typeof TONES; text: string }) {
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-semibold whitespace-nowrap ${TONES[tone]}`}>
      {text}
    </span>
  )
}

/** Our prediction vs the naive one, one row per measure (lower error is better). */
function Compare({ ours, rows }: { ours: string; rows: [string, string, string][] }) {
  return (
    <table className="w-full text-sm">
      <thead className={`text-left ${muted}`}>
        <tr>
          <th className="py-1.5 font-normal" />
          <th className="py-1.5 text-right font-normal">{ours}</th>
          <th className="py-1.5 text-right font-normal">Naive</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, a, b]) => (
          <tr key={label} className="border-t border-slate-100 dark:border-slate-800">
            <td className="py-1.5">{label}</td>
            <td className="py-1.5 text-right font-mono tabular-nums whitespace-nowrap">{a}</td>
            <td className="py-1.5 text-right font-mono tabular-nums whitespace-nowrap">{b}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ChecksView({ checks }: { checks: Checks }) {
  const m = checks.monthly.summary
  const cr = checks.crises.mae
  const gainShare = checks.monthly.rows.filter((r) => r.realised_loss < 0).length / checks.monthly.rows.length
  const signs = checks.alerts.filter((a) => a.sign_match).length
  const pt = (x: number) => `${(100 * x).toFixed(2)} pt`
  return (
    <section className={card}>
      <h2 className={head}>Predicted vs actual (each test scored once, rules fixed before the run)</h2>
      <div className="space-y-6 p-4">
        <div className="grid gap-6 lg:grid-cols-2">
          <div className={box}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">Test 1 · Real alerts: next 20 trading days</h3>
              <Verdict tone="neutral" text={`direction right ${signs} of ${checks.alerts.length}`} />
            </div>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className={`text-left ${muted}`}>
                  <tr>
                    <th className="py-1.5 font-normal">Alert</th>
                    <th className="py-1.5 text-right font-normal">Predicted</th>
                    <th className="py-1.5 text-right font-normal">Realised</th>
                    <th className="py-1.5 text-right font-normal">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {checks.alerts.map((a) => (
                    <tr key={a.window + a.ticker} className="border-t border-slate-100 dark:border-slate-800">
                      <td className="py-1.5 whitespace-nowrap">
                        <span className="font-mono font-semibold">{a.ticker}</span>{' '}
                        <span className="text-xs text-slate-500">{a.window.slice(0, 10)}</span>
                      </td>
                      <td className="py-1.5 text-right font-mono tabular-nums">{usd(a.predicted_loss)}</td>
                      <td className="py-1.5 text-right font-mono tabular-nums">{usd(a.realised_loss)}</td>
                      <td className={`py-1.5 text-right ${a.sign_match ? 'text-teal-600' : 'text-rose-600'}`}>
                        {a.sign_match ? '✓' : '✗'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className={`mt-3 ${muted}`}>
              Direction mostly right, size far too big: the full COVID episode is applied after the crash had already
              happened (spreads were falling in Apr–Jun 2020).
            </p>
          </div>

          <div className={box}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">Test 2 · 12 credit crises 1990–2023: next 60 days</h3>
              <Verdict tone="neutral" text="about the same as naive" />
            </div>
            <p className={`mt-1 ${muted}`}>
              Crisis = Baa widens 75 bp within 60 days. Prediction = the 3 most similar past crises on their first 5
              days. Naive = the average of all other crises.
            </p>
            <div className="mt-3">
              <Compare
                ours="Similar crises"
                rows={[
                  ['Loss error', usd(cr.predicted.loss), usd(cr.naive.loss)],
                  ['CET1 error', pt(cr.predicted.cet1), pt(cr.naive.cet1)],
                  ['Direction right', pct(cr.predicted.sign, 0), pct(cr.naive.sign, 0)],
                ]}
              />
            </div>
            <p className={`mt-3 ${muted}`}>No better than the average: a crisis' first week does not tell how it ends.</p>
          </div>
        </div>

        <div className={box}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold">
              Test 3 · {m.origins} month-ends 1995–2023: next 60 days from today's spreads and VIX
            </h3>
            <Verdict tone={m.passed ? 'good' : 'bad'} text={m.passed ? 'PASS' : `FAIL · ratio ${m.ratio.toFixed(2)}, pass ≤ ${m.pass_ratio}`} />
          </div>
          <div className="mt-3 grid gap-6 lg:grid-cols-3">
            <div className="min-w-0 space-y-3">
              <Compare
                ours="State model"
                rows={[
                  ['Loss error', usd(m.model_mae), usd(m.naive_mae)],
                  ['Direction right', pct(m.model_sign, 0), pct(m.naive_sign, 0)],
                ]}
              />
              <p className="text-sm">
                Direction: <span className="font-mono font-semibold text-teal-600">{pct(m.model_sign, 0)}</span> vs naive{' '}
                {pct(m.naive_sign, 0)}.{' '}
                <span className={muted}>
                  Always guessing "gain" would get {pct(gainShare, 0)}. Reported, not part of the pass rule.
                </span>
              </p>
              <p className={muted}>
                Diebold–Mariano p = {m.dm_p.toFixed(2)}: no real difference in loss error. Crashes from calm levels
                (2008, 2020) cannot be seen coming in today's spreads and VIX. That is why capital is set on severe stress
                episodes, not on a forecast.
              </p>
            </div>
            <div className="min-w-0 lg:col-span-2">
              <MonthlyChart rows={checks.monthly.rows} />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default function ModuleB() {
  const [params] = useSearchParams()
  const episode = params.get('episode') ?? 'covid2020'
  const alerts = useApi<Alert[]>(`/module-b/alerts?episode=${episode}`)
  const book = useApi<{ ticker: string }[]>('/module-b/book')

  // URL: ?doc=<alert>&ticker=, ?ticker=&notches= (what-if) or ?market=1; nothing = the first alert
  const first = alerts.data?.[0]
  const picked = params.has('market') || (!params.get('doc') && params.has('ticker'))
  const doc = picked ? null : (params.get('doc') ?? first?.doc_id ?? null)
  const query = new URLSearchParams({ episode })
  if (doc) {
    query.set('doc_id', doc)
    const ticker = params.get('doc') ? params.get('ticker') : first?.ticker
    if (ticker) query.set('ticker', ticker)
  } else if (!params.has('market') && params.get('ticker')) {
    query.set('ticker', params.get('ticker')!)
    query.set('notches', params.get('notches') ?? '1')
  }
  const ready = alerts.data != null && (doc != null || picked || alerts.data.length === 0)
  const report = useApi<Report>(ready ? `/module-b?${query}` : null)
  const checks = useApi<Checks>('/module-b/checks')
  const error = alerts.error ?? book.error ?? report.error ?? checks.error
  if (error) return <ErrorBox message={error} />
  if (!alerts.data || !book.data) return <p className="text-sm text-slate-500">Loading…</p>

  const tickers = [...new Set(book.data.map((p) => p.ticker))].sort()
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Module B · Credit stress test</h2>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          An Indian bank (capital in rupees, RBI 8% CET1 floor) lends to our 15 companies: 0.5% of each one's real debt
          from its SEC filing at 31 Dec 2019, as revolvers and bonds, with CDS and swap hedges. A credit alert cuts the
          company's rating; a real market episode moves spreads, stocks and the rupee. The book is revalued (IFRS 9,
          Basel IRB, market-risk capital) and the defaults are simulated together.
        </p>
      </div>
      <Picker alerts={alerts.data} tickers={tickers} doc={doc} />
      {report.data ? <ReportView rep={report.data} /> : <p className="text-sm text-slate-500">Running the scenario…</p>}
      {checks.data && <ChecksView checks={checks.data} />}
    </div>
  )
}
