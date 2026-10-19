import { useEffect, useMemo, useRef, useState } from 'react'
import { gsap, MOTION_OK, useGSAP } from '../motion'
import { getJSON, postJSON } from '../api'
import ErrorBox from '../components/ErrorBox'
import EventBadge, { ImpactPill } from '../components/EventBadge'
import StatCard from '../components/StatCard'
import Tickers from '../components/Tickers'
import { signed } from '../format'

type LiveSignal = {
  doc_id: string
  text: string
  url: string
  domain: string
  published_at: string
  tickers: string[]
  analysts: string[]
  sentiment: number
  event: string
  impact: number
  model: string
  live_source: string
}

type LiveState = {
  polling: boolean
  last_poll: string | null
  last_success: string | null
  error: string | null
  warning: string | null
  active_source: string | null
  poll_interval_seconds: number
  total: number
  high_impact: number
  tickers: string[]
  signals: LiveSignal[]
}

const observed = (value: string) => new Date(value).toLocaleString()

/** "12 min" / "3 h" since a time, for how fresh the newest headline is. */
const age = (value: string) => {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60000))
  return minutes < 90 ? `${minutes} min` : `${Math.round(minutes / 60)} h`
}

export default function Live() {
  const [data, setData] = useState<LiveState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [automatic, setAutomatic] = useState(true)
  const [version, setVersion] = useState(0)
  const [refreshing, setRefreshing] = useState(false)
  const [ticker, setTicker] = useState('')
  const [minImpact, setMinImpact] = useState(1)
  const [search, setSearch] = useState('')

  useEffect(() => {
    let active = true
    let timer: ReturnType<typeof setTimeout>
    const load = async () => {
      try {
        const result = await getJSON<LiveState>('/live')
        if (active) {
          setData(result)
          setError(null)
        }
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : 'Unable to load the feed')
      }
      if (active && automatic) timer = setTimeout(load, 5000)
    }
    void load()
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [automatic, version])

  const refresh = async () => {
    setRefreshing(true)
    try {
      await postJSON('/live/refresh', {})
      setVersion((value) => value + 1)
      setError(null)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Refresh failed')
    } finally {
      setRefreshing(false)
    }
  }

  const newest = data?.signals.reduce<string | null>(
    (latest, signal) => (!latest || signal.published_at > latest ? signal.published_at : latest), null)

  const shown = useMemo(() => (data?.signals ?? []).filter((signal) =>
    signal.impact >= minImpact && (!ticker || signal.tickers.includes(ticker)) &&
    `${signal.text} ${signal.domain}`.toLowerCase().includes(search.toLowerCase()),
  ), [data, ticker, minImpact, search])

  // Headlines that arrive after the first load glow for a moment, so a refresh shows what is new.
  const list = useRef<HTMLDivElement>(null)
  const seen = useRef<Set<string> | null>(null)
  useGSAP(
    () => {
      const ids = (data?.signals ?? []).map((signal) => signal.doc_id)
      if (!ids.length) return
      if (seen.current === null) {
        seen.current = new Set(ids) // the first load is the baseline, nothing is "new" yet
        return
      }
      const fresh = ids.filter((id) => !seen.current!.has(id))
      ids.forEach((id) => seen.current!.add(id))
      if (!fresh.length) return
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        const rows = fresh.map((id) => `[data-id="${CSS.escape(id)}"]`).join(',')
        gsap.fromTo(rows, { backgroundColor: 'rgba(20,184,166,0.16)' }, { backgroundColor: 'rgba(20,184,166,0)', duration: 1.8, ease: 'power2.out', clearProps: 'backgroundColor' })
      })
    },
    { scope: list, dependencies: [data] },
  )

  return (
    <div className="min-w-0 space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-medium text-teal-700 dark:text-teal-400">
            <span className={`h-2 w-2 rounded-full ${data?.error || error ? 'bg-amber-500' : 'bg-teal-500'}`} />
            Current news · GDELT
          </div>
          <h1 className="text-3xl font-semibold">Live GDELT feed</h1>
          <p className="mt-1 max-w-2xl text-sm text-slate-500 dark:text-slate-400">
            Financial headlines scored by our risk engine. New articles are checked every five minutes.
          </p>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            This feature requires an active internet connection to fetch live news.
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            <input type="checkbox" checked={automatic} onChange={(event) => setAutomatic(event.target.checked)} className="accent-teal-600" />
            Auto-refresh view
          </label>
          <button type="button" onClick={() => void refresh()} disabled={refreshing || data?.polling}
            className="rounded-lg bg-teal-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-50">
            {refreshing || data?.polling ? 'Checking news…' : 'Check for news'}
          </button>
        </div>
      </div>

      {error && <ErrorBox message={error} />}
      {data?.error && (
        <div role="status" className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
          Unable to update live news. This feature requires an active internet connection. Please connect to the internet and try again.
          <p className="mt-1 text-xs">If you are already connected, GDELT may be temporarily unavailable. We’ll retry automatically; previously fetched headlines remain visible.</p>
        </div>
      )}

      {!data?.error && (data?.warning || newest) && (
        <p role="status" className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-600 dark:text-slate-300">
          {newest && <span>Newest headline observed {age(newest)} ago.</span>}
          {data?.warning && data.active_source === 'gdelt_gkg' && (
            <span className="text-amber-800 dark:text-amber-300">
              GDELT's news search is not answering, so headlines come from its 15-minute GKG file instead.
            </span>
          )}
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Recent headlines" value={String(data?.total ?? 0)} note="latest 500 retained" />
        <StatCard label="High impact (> 7)" value={String(data?.high_impact ?? 0)} tone="alert" note="across recent headlines" />
        <StatCard label="Companies covered" value={String(data?.tickers.length ?? 0)} note="linked to our stock universe" />
        <StatCard label="Feed status" value={data?.polling ? 'Fetching' : data?.error || error ? 'Retrying' : data?.last_success ? 'Connected' : 'Starting'}
          note={data?.last_success ? `Last success: ${observed(data.last_success)}` : 'Waiting for the first poll'} />
      </div>

      <section className="min-w-0 rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="flex flex-col gap-3 border-b border-slate-200 p-4 dark:border-slate-800 lg:flex-row lg:items-center lg:justify-between">
          <h2 className="shrink-0 text-lg font-semibold">Latest risk signals <span className="font-normal text-slate-500">· {shown.length}</span></h2>
          <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:flex-wrap">
            <input aria-label="Search headlines" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search headlines or source…"
              className="min-w-0 rounded-lg border border-slate-200 bg-transparent px-3 py-2 text-sm dark:border-slate-700" />
            <select aria-label="Company" value={ticker} onChange={(event) => setTicker(event.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-900">
              <option value="">All companies & market</option>
              {data?.tickers.map((company) => <option key={company} value={company}>{company}</option>)}
            </select>
            <select aria-label="Minimum impact" value={minImpact} onChange={(event) => setMinImpact(Number(event.target.value))}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-900">
              {[1, 5, 6, 7, 8].map((impact) => <option key={impact} value={impact}>Impact ≥ {impact}</option>)}
            </select>
          </div>
        </div>
        {!shown.length && <p className="p-6 text-sm text-slate-500 dark:text-slate-400">
          {!data || data.polling ? 'Fetching and analyzing current headlines…' : data.error ? 'Waiting for GDELT to reconnect.' : 'No headlines match the current filters.'}
        </p>}
        <div ref={list} className="divide-y divide-slate-100 dark:divide-slate-800">
          {shown.map((signal) => (
            <article key={signal.doc_id} data-id={signal.doc_id} className="grid min-w-0 gap-3 p-4 sm:p-5 lg:grid-cols-[minmax(0,1fr)_auto] lg:gap-6">
              <div className="min-w-0 space-y-2">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
                  <span className="break-all font-medium">{signal.domain}</span>
                  <time dateTime={signal.published_at}>Observed {observed(signal.published_at)}</time>
                  <span>{signal.live_source === 'gdelt_gkg' ? 'GKG' : 'DOC 2.0'}</span>
                </div>
                <a href={signal.url} target="_blank" rel="noopener noreferrer" className="block break-words text-sm font-medium leading-relaxed transition hover:text-teal-600 dark:hover:text-teal-400">
                  {signal.text} <span aria-hidden className="text-slate-400">↗</span>
                </a>
                {signal.tickers.length || signal.analysts.length ? <Tickers subjects={signal.tickers} analysts={signal.analysts} /> : <span className="text-xs text-slate-500">Market-wide / no linked company</span>}
              </div>
              <div className="flex flex-wrap items-center gap-3 lg:justify-end" title={`Model: ${signal.model}`}>
                <EventBadge event={signal.event} />
                <span className="flex items-center gap-1.5 text-xs text-slate-500">Impact <ImpactPill impact={signal.impact} /></span>
                <span className={`font-figures text-sm tabular-nums ${signal.sentiment > 0.2 ? 'text-teal-700 dark:text-teal-400' : signal.sentiment < -0.2 ? 'text-rose-600 dark:text-rose-400' : 'text-slate-500'}`}>
                  <span className="mr-1 font-sans text-xs text-slate-500">Sentiment</span>{signed(signal.sentiment, 3)}
                </span>
              </div>
            </article>
          ))}
        </div>
      </section>
      <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Headline analysis only. Times show when GDELT observed each article. Live signals are stored separately from historical replay and backtests.
        {' '}Manual checks are spaced at least one minute apart.
      </p>
    </div>
  )
}
