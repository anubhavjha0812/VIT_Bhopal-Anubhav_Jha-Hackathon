import { useSearchParams } from 'react-router-dom'
import { useApi } from '../api'
import Chart from '../components/Chart'
import ErrorBox from '../components/ErrorBox'
import PageHero from '../components/PageHero'
import StatCard from '../components/StatCard'
import { chartColors, useTheme } from '../theme'

// Shapes of /module-b, /module-b/alerts and /module-b/checks (src/riskengine/api.py).
interface Capital {
  cet1_before: number
  cet1_after: number
  floor: number
  loss_usd: number
  rwa_before_inr: number
  capital_before_inr: number
  capital_after_inr: number
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
  top_losses: {
    position: string
    ticker: string
    instrument: string
    rating: string
    notional: number
    loss: number
  }[]
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
  by_class: {
    instrument: string
    notional: number
    value_before: number
    value_after: number
    loss: number
  }[]
  cet1_bridge: { before: number; loss: number; rwa: number; after: number }
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
interface Paths {
  paths: number
  sigma: number
  seed: number
  histogram: { edges: number[]; counts: number[] }
  median_loss: number
  p5_loss: number
  p1_loss: number
  es975_loss: number
  cet1_p1: number
  cet1_median: number
  p_below_floor: number
}
interface Checks {
  alerts: {
    ticker: string
    window: string
    predicted_loss: number
    naive_loss: number
    realised_loss: number
    sign_match: boolean
  }[]
  crises: {
    mae: Record<'predicted' | 'naive', { loss: number; cet1: number; sign: number }>
    rows: unknown[]
  }
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
    rows: {
      origin: string
      realised_loss: number
      model_loss: number
      naive_loss: number
    }[]
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
        <span className="font-figures">{pct(c.cet1_before)}</span> (black line) →{' '}
        <span className={`font-figures font-semibold ${breach ? 'text-rose-600 dark:text-rose-400' : ''}`}>
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

const investmentGrade = (r: string) => /^(A|BBB)/.test(r)

/** Waterfall bars: an invisible base plus the visible step, ECharts' usual way. */
function waterfall(steps: { name: string; value: number; total?: boolean }[]) {
  let run = 0
  return steps.map((st) => {
    if (st.total) return { base: Math.min(0, st.value), size: Math.abs(st.value), ...st }
    const from = run
    run += st.value
    return { base: Math.min(from, run), size: Math.abs(st.value), ...st }
  })
}

/** Where the loss comes from: each product's loss, then the total. */
function LossWaterfall({ rows, total }: { rows: Report['by_class']; total: number }) {
  const c = chartColors(useTheme())
  const sorted = [...rows].sort((a, b) => b.loss - a.loss)
  const bars = waterfall([
    ...sorted.map((r) => ({
      name: INSTRUMENTS[r.instrument] ?? r.instrument,
      value: r.loss / 1e6,
    })),
    { name: 'Total loss', value: total / 1e6, total: true },
  ])
  const notional = Object.fromEntries(sorted.map((r) => [INSTRUMENTS[r.instrument] ?? r.instrument, r.notional]))
  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: { left: 110, right: 90, top: 8, bottom: 28 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (ps: { dataIndex: number }[]) => {
        const b = bars[ps[0].dataIndex]
        const n = notional[b.name]
        return `${b.name}<br/>${b.value >= 0 ? 'loss' : 'gain'} $${Math.abs(b.value).toFixed(1)}m${
          n ? `<br/>notional $${(n / 1e6).toFixed(0)}m` : ''
        }`
      },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: bars.map((b) => b.name),
      axisLabel: { color: c.text },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: c.grid } },
    },
    xAxis: {
      type: 'value',
      axisLabel: { color: c.text, formatter: (v: number) => `$${v}m` },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: [
      {
        type: 'bar',
        stack: 'w',
        silent: true,
        itemStyle: { color: 'transparent' },
        data: bars.map((b) => b.base),
      },
      {
        type: 'bar',
        stack: 'w',
        barWidth: '55%',
        data: bars.map((b) => ({
          value: b.size,
          itemStyle: {
            color: b.value >= 0 ? c.down : c.up,
            opacity: b.total ? 1 : 0.85,
            borderRadius: 3,
          },
        })),
        label: {
          show: true,
          position: 'right',
          color: c.text,
          formatter: (p: { dataIndex: number }) => {
            const v = bars[p.dataIndex].value
            return `${v >= 0 ? '−' : '+'}$${Math.abs(v).toFixed(1)}m`
          },
        },
      },
    ],
  }
  return <Chart option={option} style={{ height: 40 * bars.length + 40 }} />
}

/** CET1 before → after in two exact steps (after-tax loss, RWA change), against the floor. */
function Cet1Bridge({ b, floor }: { b: Report['cet1_bridge']; floor: number }) {
  const c = chartColors(useTheme())
  const bars = waterfall([
    { name: 'Before', value: 100 * b.before, total: true },
    { name: 'Loss after tax', value: 100 * b.loss },
    { name: 'RWA change', value: 100 * b.rwa },
    { name: 'After', value: 100 * b.after, total: true },
  ])
  const colour = (x: (typeof bars)[number]) =>
    x.total
      ? x.name === 'After' && b.after < floor
        ? c.down
        : x.name === 'After'
          ? c.line
          : c.text
      : x.value < 0
        ? c.down
        : c.up
  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: { left: 44, right: 16, top: 24, bottom: 28 },
    tooltip: { show: false },
    xAxis: {
      type: 'category',
      data: bars.map((x) => x.name),
      axisLabel: { color: c.text },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      axisLabel: { color: c.text, formatter: (v: number) => `${v}%` },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: [
      {
        type: 'bar',
        stack: 'w',
        silent: true,
        itemStyle: { color: 'transparent' },
        data: bars.map((x) => x.base),
      },
      {
        type: 'bar',
        stack: 'w',
        barWidth: '50%',
        data: bars.map((x) => ({
          value: x.size,
          itemStyle: {
            color: colour(x),
            opacity: x.total ? 0.9 : 0.8,
            borderRadius: 3,
          },
        })),
        label: {
          show: true,
          position: 'top',
          color: c.text,
          formatter: (p: { dataIndex: number }) => {
            const x = bars[p.dataIndex]
            return x.total ? `${x.value.toFixed(2)}%` : `${x.value >= 0 ? '+' : '−'}${Math.abs(x.value).toFixed(2)} pt`
          },
        },
        markLine: {
          symbol: 'none',
          silent: true,
          data: [{ yAxis: 100 * floor }],
          lineStyle: { color: c.down, type: 'dashed' },
          label: {
            formatter: `floor ${(100 * floor).toFixed(0)}%`,
            color: c.down,
            position: 'insideEndTop',
          },
        },
      },
    ],
  }
  return <Chart option={option} style={{ height: 280 }} />
}

/** Capital and credit facts for the scenario, before → after. */
function CapitalTable({ rep }: { rep: Report }) {
  const { capital: c, obligors, stages } = rep
  const cr = (inr: number) => `₹${(inr / 1e7).toLocaleString(undefined, { maximumFractionDigits: 0 })} cr`
  const moved = obligors.filter((o) => o.rating_before !== o.rating_after)
  const fallen = moved.filter((o) => investmentGrade(o.rating_before) && !investmentGrade(o.rating_after))
  const stressed = (stages['2'] ?? 0) + (stages['3'] ?? 0)
  const rows: [string, string, boolean?][] = [
    ['CET1 ratio', `${pct(c.cet1_before)} → ${pct(c.cet1_after)}`, c.cet1_after < c.floor],
    ['CET1 capital', `${cr(c.capital_before_inr)} → ${cr(c.capital_after_inr)}`],
    ['Risk-weighted assets', `${cr(c.rwa_before_inr)} → ${cr(c.rwa_after_inr)}`],
    [
      'Headroom vs floor',
      `${Math.round(1e4 * (c.cet1_before - c.floor))} bp → ${Math.round(1e4 * (c.cet1_after - c.floor))} bp`,
      c.cet1_after < c.floor,
    ],
    ['USD/INR', `${c.fx_before.toFixed(2)} → ${c.fx_after.toFixed(2)}`],
    ['Revolvers in Stage 2/3', `0 → ${stressed} of ${Object.values(stages).reduce((a, b) => a + b, 0)}`],
    ['Downgraded names', `${moved.length} · ${fallen.length} fallen angel${fallen.length === 1 ? '' : 's'}`],
  ]
  return (
    <table className="w-full text-sm">
      <tbody>
        {rows.map(([k, v, bad]) => (
          <tr key={k} className="border-b border-slate-100 last:border-0 dark:border-slate-800">
            <td className="py-2 text-slate-600 dark:text-slate-400">{k}</td>
            <td
              className={`py-2 text-right font-figures tabular-nums whitespace-nowrap ${
                bad ? 'font-semibold text-rose-600 dark:text-rose-400' : ''
              }`}
            >
              {v}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Shock-size uncertainty: the same scenario at 500 lognormal multiples of its size. */
function ShockPaths({ s, loss, floor }: { s: Report['scenario']; loss: number; floor: number }) {
  const c = chartColors(useTheme())
  const q = new URLSearchParams({
    episode: s.episode,
    multiple: String(s.multiple),
    notches: String(s.notches),
  })
  if (s.ticker) q.set('ticker', s.ticker)
  const paths = useApi<Paths>(`/module-b/paths?${q}`)
  if (paths.error) return <p className={`p-4 ${muted}`}>{paths.error}</p>
  if (!paths.data) return <p className={`p-4 ${muted}`}>Revaluing the book on 21 shock sizes…</p>
  const p = paths.data
  const { edges, counts } = p.histogram
  const mid = edges.slice(1).map((e, i) => (e + edges[i]) / 2 / 1e6)
  const at = (x: number) => Math.max(0, Math.min(mid.length - 1, edges.findIndex((e) => e > x) - 1))
  const mark = (name: string, x: number, color: string) => ({
    xAxis: at(x),
    lineStyle: {
      color,
      type: name === 'scenario' ? 'solid' : 'dashed',
      width: 1.5,
    },
    label: { formatter: name, color, position: 'end' },
  })
  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: { left: 40, right: 16, top: 28, bottom: 40 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (ps: { dataIndex: number; value: number }[]) =>
        `loss ≈ $${mid[ps[0].dataIndex].toFixed(0)}m<br/>${ps[0].value} paths`,
    },
    xAxis: {
      type: 'category',
      data: mid.map((m) => m.toFixed(0)),
      name: 'loss before tax, $m',
      nameLocation: 'middle',
      nameGap: 26,
      nameTextStyle: { color: c.text },
      axisLabel: { color: c.text, interval: 7 },
      axisLine: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: c.text },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: [
      {
        type: 'bar',
        barCategoryGap: '8%',
        itemStyle: { color: c.down, opacity: 0.8, borderRadius: [2, 2, 0, 0] },
        data: counts,
        markLine: {
          symbol: 'none',
          silent: true,
          data: [mark('scenario', loss, c.line), mark('P5', p.p5_loss, '#d99a00'), mark('P1', p.p1_loss, c.down)],
        },
      },
    ],
  }
  const tiles: [string, string, string, boolean?][] = [
    ['P5 loss (1-in-20)', usd(p.p5_loss), `median ${usd(p.median_loss)}`],
    ['P1 loss (1-in-100)', usd(p.p1_loss), `ES 97.5% ${usd(p.es975_loss)}`],
    ['CET1 at P1', pct(p.cet1_p1), `median ${pct(p.cet1_median)}`, p.cet1_p1 < floor],
    ['P(CET1 < floor)', pct(p.p_below_floor, 1), `floor ${pct(floor, 0)}`, p.p_below_floor > 0],
  ]
  return (
    <div className="space-y-3 p-4">
      <Chart option={option} style={{ height: 240 }} />
      <div className="grid grid-cols-2 gap-3">
        {tiles.map(([k, v, note, bad]) => (
          <div key={k} className={box}>
            <div className={muted}>{k}</div>
            <div className={`font-figures text-2xl font-semibold ${bad ? 'text-rose-600 dark:text-rose-400' : ''}`}>
              {v}
            </div>
            <div className={muted}>{note}</div>
          </div>
        ))}
      </div>
      <p className={muted}>
        {p.paths} paths: the episode size × a mean-one lognormal draw (σ {p.sigma}, seed {p.seed}). The book is fully
        revalued at 21 sizes; each path is read off that curve.
      </p>
    </div>
  )
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
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v: number) => `$${v.toFixed(1)}m`,
    },
    xAxis: {
      type: 'time',
      axisLabel: { color: c.text },
      axisLine: { lineStyle: { color: c.grid } },
    },
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
          <div className="grid gap-2 lg:grid-cols-2">
            {alerts.map((a) => (
              <button
                key={`${a.doc_id}-${a.ticker}`}
                onClick={() =>
                  set({
                    doc: a.doc_id,
                    ticker: a.ticker,
                    notches: null,
                    market: null,
                  })
                }
                className={`rounded-lg border p-3 text-left text-sm ${
                  doc === a.doc_id
                    ? 'border-teal-600 bg-teal-50 dark:bg-teal-950/40'
                    : 'border-slate-200 hover:border-slate-400 dark:border-slate-700'
                }`}
              >
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-figures font-semibold">{a.ticker}</span>
                  <span className="text-slate-500">
                    {a.published_at.slice(0, 10)} · {a.source}
                  </span>
                  <span className="w-full font-figures sm:ml-auto sm:w-auto">CET1 {pct(a.cet1_after)}</span>
                </div>
                <div className="mt-1 line-clamp-2">{a.text}</div>
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-end">
          <button
            className={`${pill(marketOnly)} w-full sm:w-auto`}
            onClick={() => set({ doc: null, ticker: null, notches: null, market: '1' })}
          >
            Market only
          </button>
          <label className="text-sm">
            <span className={`block ${muted}`}>What-if: company</span>
            <select
              value={whatIf ? (params.get('ticker') ?? '') : ''}
              onChange={(e) =>
                set(
                  e.target.value
                    ? {
                        doc: null,
                        market: null,
                        ticker: e.target.value,
                        notches: params.get('notches') ?? '1',
                      }
                    : { ticker: null, market: '1' },
                )
              }
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 dark:border-slate-700 dark:bg-slate-900 sm:w-auto"
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
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 disabled:opacity-40 dark:border-slate-700 dark:bg-slate-900 sm:w-auto"
            >
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <option key={n}>{n}</option>
              ))}
              <option value="99">default</option>
            </select>
          </label>
          <div className="w-full sm:ml-auto sm:w-auto">
            <span className={`block ${muted}`}>Market episode (real FRED and price moves)</span>
            <div className="mt-1 flex flex-wrap gap-2">
              {EPISODES.map(([key, label]) => (
                <button
                  key={key}
                  className={`${pill(episode === key)} flex-1 sm:flex-none`}
                  onClick={() => set({ episode: key })}
                >
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
      <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-4">
        <StatCard
          hero
          label="CET1 ratio"
          value={`${pct(c.cet1_before, 1)} → ${pct(c.cet1_after, 1)}`}
          note={`floor ${pct(c.floor, 0)}`}
          tone={c.cet1_after < c.floor ? 'alert' : 'default'}
        />
        <StatCard hero label="Loss before tax" value={usd(c.loss_usd)} note={`book of ${rep.book_date}`} />
        <StatCard
          hero
          label={`ES ${pct(mc.level, 0)} (defaults)`}
          value={`${usd(mc.before.es)} → ${usd(mc.after.es)}`}
          note={`${mc.draws.toLocaleString()} correlated draws`}
        />
        <StatCard
          hero
          label="Reverse stress"
          value={rep.reverse_stress == null ? '> 20×' : `${rep.reverse_stress.toFixed(2)}×`}
          note={`of ${s.episode_label} takes CET1 to ${pct(c.floor, 0)}`}
        />
      </div>

      <div className="grid min-w-0 gap-6 pt-2 xl:grid-cols-5">
        <section className={`${card} xl:col-span-3`}>
          <h2 className={head}>Risk memo</h2>
          <div className="space-y-4 p-4">
            <Cet1Gauge c={c} />
            <Memo text={rep.memo} />
            <p className={muted}>Fixed template, no LLM: every number above comes from this run.</p>
          </div>
        </section>
        <section className={`${card} min-w-0 xl:col-span-2`}>
          <h2 className={head}>Top 10 position losses</h2>
          <div className="overflow-x-auto">
            <table className="min-w-[520px] w-full text-sm">
              <thead className="text-left text-xs text-slate-500 dark:text-slate-400">
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
                    <td className="px-4 py-2 font-figures">
                      {p.ticker} <span className="text-xs text-slate-500">{p.rating}</span>
                    </td>
                    <td className="px-4 py-2">{INSTRUMENTS[p.instrument] ?? p.instrument}</td>
                    <td className="px-4 py-2 text-right font-figures tabular-nums">{usd(p.notional)}</td>
                    <td
                      className={`px-4 py-2 text-right font-figures tabular-nums ${p.loss < 0 ? 'text-teal-600' : ''}`}
                    >
                      {usd(p.loss)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div
            className={`flex flex-wrap gap-x-4 gap-y-1 border-t border-slate-200 px-4 py-3 dark:border-slate-800 ${muted}`}
          >
            {Object.entries(rep.by_instrument).map(([k, v]) => (
              <span key={k}>
                {INSTRUMENTS[k] ?? k}: <span className="font-figures">{usd(v)}</span>
              </span>
            ))}
          </div>
        </section>
      </div>

      <div className="grid min-w-0 gap-6 pt-2 xl:grid-cols-2">
        <section className={`${card} min-w-0`}>
          <h2 className={head}>Loss by product</h2>
          <div className="p-2">
            <LossWaterfall rows={rep.by_class} total={c.loss_usd} />
          </div>
          <p className={`px-4 pb-3 ${muted}`}>
            Each product's loss before tax, then the total. Green bars are gains (bought CDS protection, pay-fixed swaps
            when rates rise).
          </p>
        </section>
        <section className={`${card} min-w-0`}>
          <h2 className={head}>Shock-size uncertainty · Monte Carlo</h2>
          <ShockPaths s={s} loss={c.loss_usd} floor={c.floor} />
        </section>
      </div>

      <section className={`${card} mt-2`}>
        <h2 className={head}>Capital impact · CET1 bridge</h2>
        <div className="grid gap-6 p-4 lg:grid-cols-2">
          <Cet1Bridge b={rep.cet1_bridge} floor={c.floor} />
          <CapitalTable rep={rep} />
        </div>
      </section>

      <div className="grid min-w-0 gap-6 pt-2 xl:grid-cols-5">
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
          <div className="max-h-96 overflow-y-auto overflow-x-hidden">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white text-left text-xs text-slate-500 dark:bg-slate-900 dark:text-slate-400">
                <tr>
                  <th className="py-2 pl-4 pr-2">Name</th>
                  <th className="px-2 py-2">Rating</th>
                  <th className="px-2 py-2 text-right">Spread bp</th>
                  <th className="px-2 py-2 text-right">PD</th>
                  <th className="py-2 pl-2 pr-4 text-right">Loss</th>
                </tr>
              </thead>
              <tbody>
                {[...rep.obligors]
                  .sort((a, b) => b.loss - a.loss)
                  .map((o) => (
                    <tr key={o.ticker} className="border-t border-slate-100 dark:border-slate-800">
                      <td className="py-1.5 pl-4 pr-2 font-figures">{o.ticker}</td>
                      <td className="px-2 py-1.5 font-figures text-xs">
                        {o.rating_before === o.rating_after
                          ? o.rating_before
                          : `${o.rating_before} → ${o.rating_after}`}
                      </td>
                      <td className="px-2 py-1.5 text-right font-figures tabular-nums">
                        {Math.round(100 * o.spread_before)} → {Math.round(100 * o.spread_after)}
                      </td>
                      <td className="px-2 py-1.5 text-right font-figures tabular-nums">
                        {pct(o.pd_before)} → {pct(o.pd_after)}
                      </td>
                      <td className="py-1.5 pl-2 pr-4 text-right font-figures tabular-nums">{usd(o.loss)}</td>
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
  return <span className={`rounded px-2 py-0.5 text-xs font-semibold whitespace-nowrap ${TONES[tone]}`}>{text}</span>
}

/** Our prediction vs the naive one, one row per measure (lower error is better). */
function Compare({ ours, rows }: { ours: string; rows: [string, string, string][] }) {
  return (
    <table className="min-w-[360px] w-full text-sm">
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
            <td className="py-1.5 text-right font-figures tabular-nums whitespace-nowrap">{a}</td>
            <td className="py-1.5 text-right font-figures tabular-nums whitespace-nowrap">{b}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Sample size, period and baseline of a validation test. */
function TestFacts({ n, period, baseline }: { n: string; period: string; baseline: string }) {
  return (
    <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600 dark:text-slate-400">
      {(
        [
          ['n', n],
          ['Period', period],
          ['Baseline', baseline],
        ] as const
      ).map(([k, v]) => (
        <div key={k} className="flex gap-1">
          <dt className="font-semibold">{k}:</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

const SENSITIVITY: [string, string][] = [
  ['No shock, no loss', 'no alert and the episode × 0 leave every rating, spread, PD and CET1 unchanged'],
  [
    'Bigger shock, bigger loss',
    'GFC 2008, euro 2011 and COVID 2020 at × 0, 0.5, 1, 1.5, 2: loss never falls, CET1 never rises',
  ],
  ['× 0 removes the episode', "with a downgrade, only the obligor's rating moves; others move by contagion only"],
  ['More notches, more risk', "0 to 6 notches: the obligor's PD and ECL, total loss and RWA never fall"],
  ['Stressed PD', 'Vasicek PD rises with the market factor z, never below the normal PD'],
  ['Default', 'rating D: the loan in IFRS 9 stage 3 at LGD × EAD, the bond at its recovery value'],
  ['Reverse stress', 'at the multiple it reports, CET1 lands on the 8% floor'],
]

/** M1: do the formulas move the right way? Expectations fixed before the checks were run. */
function SensitivityView() {
  return (
    <section className={card}>
      <h2 className={head}>Stress-test checks · do the formulas move the right way?</h2>
      <div className="space-y-3 p-4">
        <p className={muted}>
          Seven checks on the real book and market data, expectations written down before they were run (DECISIONS M1);
          all pass, as 13 automated tests (tests/test_module_b_sensitivity.py).
        </p>
        <ul className="grid gap-2 sm:grid-cols-2">
          {SENSITIVITY.map(([name, what]) => (
            <li key={name} className={`${box} flex gap-3`}>
              <span className="text-teal-600" aria-hidden>
                ✓
              </span>
              <span className="text-sm">
                <span className="font-semibold">{name}:</span> {what}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
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
      <h2 className={head}>Forecast validation · does a scenario predict what really happened?</h2>
      <div className="space-y-6 p-4">
        <p className={muted}>
          A stress test asks "how bad if this happens?", not "what will happen?". These three tests ask the second
          question anyway, to show what the scenarios can and cannot tell. Each was scored once, with its rules fixed
          before the run.
        </p>
        <div className="grid gap-6 lg:grid-cols-2">
          <div className={box}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">Test 1 · Real alerts: next 20 trading days</h3>
              <Verdict tone="neutral" text={`direction right ${signs} of ${checks.alerts.length}`} />
            </div>
            <TestFacts
              n={`${checks.alerts.length} gated alerts`}
              period="Apr–Jun 2020"
              baseline="COVID episode alone, no downgrade"
            />
            <p className={`mt-2 ${muted}`}>
              Predicted = the alert's downgrade + the full COVID 2020 episode. Realised = the same downgrade + the real
              market moves of the next 20 trading days. A historical crisis shock against realised P&L: not a measure of
              forecast accuracy.
            </p>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-[560px] w-full text-sm">
                <thead className={`text-left ${muted}`}>
                  <tr>
                    <th className="py-1.5 font-normal">Alert</th>
                    <th className="py-1.5 text-right font-normal">Predicted</th>
                    <th className="py-1.5 text-right font-normal">Naive</th>
                    <th className="py-1.5 text-right font-normal">Realised</th>
                    <th className="py-1.5 text-right font-normal">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {checks.alerts.map((a) => (
                    <tr key={a.window + a.ticker} className="border-t border-slate-100 dark:border-slate-800">
                      <td className="py-1.5 whitespace-nowrap">
                        <span className="font-figures font-semibold">{a.ticker}</span>{' '}
                        <span className="text-xs text-slate-500">{a.window.slice(0, 10)}</span>
                      </td>
                      <td className="py-1.5 text-right font-figures tabular-nums">{usd(a.predicted_loss)}</td>
                      <td className="py-1.5 text-right font-figures tabular-nums">{usd(a.naive_loss)}</td>
                      <td className="py-1.5 text-right font-figures tabular-nums">{usd(a.realised_loss)}</td>
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
              happened (spreads were falling in Apr–Jun 2020). Predicted − naive is the alert's own part (about $8–10m);
              the gap to realised is the episode.
            </p>
          </div>

          <div className={box}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">Test 2 · 12 credit crises 1990–2023: next 60 days</h3>
              <Verdict tone="neutral" text="about the same as naive" />
            </div>
            <TestFacts n="12 crises (leave one out)" period="1990–2023" baseline="average of the other 11 crises" />
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
            <p className={`mt-3 ${muted}`}>
              No better than the average: a crisis' first week does not tell how it ends.
            </p>
          </div>
        </div>

        <div className={box}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold">
              Test 3 · {m.origins} month-ends 1995–2023: next 60 days from today's spreads and VIX
            </h3>
            <Verdict
              tone={m.passed ? 'good' : 'bad'}
              text={m.passed ? 'PASS' : `FAIL · ratio ${m.ratio.toFixed(2)}, pass ≤ ${m.pass_ratio}`}
            />
          </div>
          <TestFacts
            n={`${m.origins} month-ends`}
            period="1995–2023"
            baseline="average 60-day move of all earlier months"
          />
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
                Direction: <span className="font-figures font-semibold text-teal-600">{pct(m.model_sign, 0)}</span> vs
                naive {pct(m.naive_sign, 0)}.{' '}
                <span className={muted}>
                  Always guessing "gain" would get {pct(gainShare, 0)}. Reported, not part of the pass rule.
                </span>
              </p>
              <p className={muted}>
                Diebold–Mariano p = {m.dm_p.toFixed(2)}: no real difference in loss error. Crashes from calm levels
                (2008, 2020) cannot be seen coming in today's spreads and VIX. That is why capital is set on severe
                stress episodes, not on a forecast.
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

// The memo's keys ("Trigger (…):", "Scenario:", "Gate:", …) in bold, so it scans easily.
const MEMO_KEY = /(^[^:."]{1,40}:|Credit rules matched:|Gate:)/g

function Memo({ text }: { text: string }) {
  return (
    <div className="space-y-3 text-sm leading-relaxed">
      {text.split('\n\n').map((para, i) => (
        <p key={i}>
          {para.split(MEMO_KEY).map((part, j) =>
            j % 2 ? (
              <strong key={j} className="font-semibold text-slate-900 dark:text-white">
                {part}
              </strong>
            ) : (
              part
            ),
          )}
        </p>
      ))}
    </div>
  )
}

/** Shown while a scenario is revalued, simulated and reverse-stressed (a few seconds). */
function Running() {
  return (
    <div
      role="status"
      className="mx-3 flex max-w-[calc(100vw-24px)] items-center gap-3 rounded-2xl border border-teal-500/30 bg-white/90 px-4 py-2 text-sm shadow-lg backdrop-blur dark:bg-slate-900/90 sm:rounded-full"
    >
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-teal-500/30 border-t-teal-500" />
      <span className="min-w-0">Running the scenario: revaluing the book, 20,000 default draws, reverse stress…</span>
    </div>
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
      <PageHero
        title="Module B · Portfolio stress test"
        wide
        text={
          <>
            An Indian bank (capital in rupees, RBI 8% CET1 floor) lends to our 15 companies: 0.5% of each one's real
            debt from its SEC filing at 31 Dec 2019, as revolvers and bonds, with CDS and swap hedges. A credit alert
            cuts the company's rating; a real market episode moves spreads, stocks and the rupee. The book is revalued
            (IFRS 9, Basel IRB, market-risk capital) and the defaults are simulated together.
          </>
        }
      />
      <Picker alerts={alerts.data} tickers={tickers} doc={doc} />
      {report.data ? (
        <div className="relative" aria-busy={report.loading}>
          {/* a new choice is running: the last result stays, dimmed, under a loader */}
          <div
            className={`transition duration-300 ${report.loading ? 'pointer-events-none opacity-40 blur-[1px]' : ''}`}
          >
            <ReportView rep={report.data} />
          </div>
          {report.loading && (
            <div className="absolute inset-x-0 top-6 z-10 flex justify-center">
              <Running />
            </div>
          )}
        </div>
      ) : (
        <div className="flex justify-center py-10">
          <Running />
        </div>
      )}
      <SensitivityView />
      {checks.data && <ChecksView checks={checks.data} />}
    </div>
  )
}
