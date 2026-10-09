import { useCallback, useEffect, useRef, useState } from 'react'
import { signed } from '../format'
import { useSearchParams } from 'react-router-dom'
import { useApi } from '../api'
import ErrorBox from '../components/ErrorBox'
import EventBadge, { ImpactPill } from '../components/EventBadge'
import WeightBars from '../components/WeightBars'
import Tickers from '../components/Tickers'
import type { Overview } from '../types'

interface ReplaySignal {
  doc_id: string
  published_at: string
  source: string
  text: string
  tickers: string[]
  analysts: string[]
  sentiment: number
  event: string
  impact: number
  obligors?: string[] // on alerts: companies the gate passed to Module B
  gate?: string
}

interface WeightsEvent {
  date: string
  weights: Record<string, number>
}

const SPEEDS = [2, 10, 50, 200]
const FEED_SIZE = 40
type Status = 'idle' | 'playing' | 'paused' | 'done' | 'error'

const nyTime = (iso: string) =>
  new Date(iso).toLocaleString('en-US', {
    timeZone: 'America/New_York',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })

/** Replays the stored window through /api/replay (Server-Sent Events). */
function useReplay() {
  const source = useRef<EventSource | null>(null)
  const seen = useRef(new Set<string>())
  const last = useRef<string | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [feed, setFeed] = useState<ReplaySignal[]>([])
  const [alerts, setAlerts] = useState<ReplaySignal[]>([])
  const [weights, setWeights] = useState<WeightsEvent | null>(null)
  const [count, setCount] = useState({ texts: 0, news: 0, tweets: 0 })

  const close = () => {
    source.current?.close()
    source.current = null
  }

  const play = useCallback((rate: number) => {
    close()
    const params = new URLSearchParams({ rate: String(rate) })
    if (last.current) params.set('after', last.current) // resume where we paused
    const es = new EventSource(`/api/replay?${params}`)
    source.current = es
    setStatus('playing')

    es.addEventListener('signal', (e) => {
      const s: ReplaySignal = JSON.parse((e as MessageEvent).data)
      if (seen.current.has(s.doc_id)) return // resent on resume
      seen.current.add(s.doc_id)
      last.current = s.published_at
      setFeed((f) => [s, ...f].slice(0, FEED_SIZE))
      setCount((c) => ({ texts: c.texts + 1, news: c.news + (s.source === 'news' ? 1 : 0), tweets: c.tweets + (s.source === 'news' ? 0 : 1) }))
    })
    es.addEventListener('alert', (e) => {
      const s: ReplaySignal = JSON.parse((e as MessageEvent).data)
      setAlerts((a) => (a.some((x) => x.doc_id === s.doc_id) ? a : [s, ...a]))
    })
    es.addEventListener('weights', (e) => setWeights(JSON.parse((e as MessageEvent).data)))
    es.addEventListener('end', () => {
      close() // otherwise EventSource reconnects and starts again
      setStatus('done')
    })
    es.onerror = () => {
      if (source.current === es) {
        close()
        setStatus('error')
      }
    }
  }, [])

  const pause = () => {
    close()
    setStatus('paused')
  }

  const reset = () => {
    close()
    seen.current = new Set()
    last.current = null
    setFeed([])
    setAlerts([])
    setWeights(null)
    setCount({ texts: 0, news: 0, tweets: 0 })
    setStatus('idle')
  }

  useEffect(() => close, []) // stop streaming when leaving the page
  return { status, feed, alerts, weights, count, play, pause, reset }
}

const card = 'rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900'
const head = 'border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800'

export default function Replay() {
  const overview = useApi<Overview>('/overview')
  const [params] = useSearchParams()
  const [rate, setRate] = useState(() => Number(params.get('rate')) || 10)
  const r = useReplay()
  const { play } = r
  const autoplay = params.get('autoplay') === '1' // demo links: /replay?autoplay=1&rate=50
  useEffect(() => {
    if (autoplay) play(rate)
    // start once on load; later speed changes go through the select
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoplay, play])

  if (overview.error) return <ErrorBox message={overview.error} />
  const total = overview.data?.texts ?? 0
  const trigger = overview.data?.config.impact_trigger ?? 7
  const now = r.feed[0]?.published_at

  return (
    <div className="space-y-6">
      <section className={`${card} flex flex-wrap items-center gap-4 p-4`}>
        {r.status === 'playing' ? (
          <button type="button" onClick={r.pause} className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white dark:bg-slate-200 dark:text-slate-900">
            ❚❚ Pause
          </button>
        ) : (
          <button
            type="button"
            onClick={() => r.play(rate)}
            disabled={r.status === 'done'}
            className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-700 disabled:opacity-50"
          >
            ▶ {r.status === 'paused' ? 'Resume' : 'Play'}
          </button>
        )}
        <button type="button" onClick={r.reset} className="rounded-lg border border-slate-200 px-3 py-2 text-sm dark:border-slate-700">
          Reset
        </button>
        <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
          Speed
          <select
            value={rate}
            onChange={(e) => {
              const v = Number(e.target.value)
              setRate(v)
              if (r.status === 'playing') r.play(v)
            }}
            className="rounded border border-slate-200 bg-white px-2 py-1 dark:border-slate-700 dark:bg-slate-950"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s} texts/s
              </option>
            ))}
          </select>
        </label>
        <div className="ml-auto text-right">
          <div className="font-mono text-lg font-semibold tabular-nums">{now ? `${nyTime(now)} NY` : '—'}</div>
          <div className="text-xs text-slate-500">
            {r.count.texts.toLocaleString()} / {total.toLocaleString()} texts · {r.count.news} news · {r.count.tweets} tweets
            {r.status === 'done' && ' · finished'}
            {r.status === 'error' && ' · connection lost (is the API running?)'}
          </div>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
          <div className="h-full bg-teal-500 transition-[width]" style={{ width: total ? `${(r.count.texts / total) * 100}%` : 0 }} />
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-3">
        <section className={`${card} xl:col-span-2`}>
          <h2 className={head}>Live feed · newest first</h2>
          {r.feed.length === 0 ? (
            <p className="p-4 text-sm text-slate-500">
              Press Play: the 9 Apr – 10 Jun 2020 news and tweets arrive in the order they were published, each scored by
              the engine.
            </p>
          ) : (
            <ul>
              {r.feed.map((s) => (
                <li key={s.doc_id} className="flex items-start gap-3 border-t border-slate-100 px-4 py-2 text-sm first:border-t-0 dark:border-slate-800">
                  <span className="w-24 shrink-0 font-mono text-xs text-slate-500">{nyTime(s.published_at)}</span>
                  <span className="w-28 shrink-0">
                    <Tickers subjects={s.tickers} analysts={s.analysts} />
                  </span>
                  <span
                    className={`w-12 shrink-0 text-right font-mono tabular-nums ${
                      s.sentiment > 0.2 ? 'text-teal-700 dark:text-teal-400' : s.sentiment < -0.2 ? 'text-rose-600 dark:text-rose-400' : ''
                    }`}
                  >
                    {signed(s.sentiment)}
                  </span>
                  <ImpactPill impact={s.impact} trigger={trigger} />
                  <span className="min-w-0 flex-1">
                    {s.event !== 'Other' && (
                      <span className="mr-2">
                        <EventBadge event={s.event} />
                      </span>
                    )}
                    {s.text}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <div className="space-y-6">
          <section className={card}>
            <h2 className={head}>
              Credit alerts · Credit Chhanni <span className="font-normal text-rose-600">({r.alerts.length})</span>
            </h2>
            {r.alerts.length === 0 ? (
              <p className="p-4 text-sm text-slate-500">None yet.</p>
            ) : (
              <ul className="max-h-80 overflow-y-auto">
                {r.alerts.map((s) => (
                  <li key={s.doc_id} className="border-t border-slate-100 px-4 py-2 text-sm first:border-t-0 dark:border-slate-800">
                    <div className="flex items-center gap-2 text-xs">
                      <ImpactPill impact={s.impact} trigger={trigger} />
                      <EventBadge event={s.event} />
                      <Tickers subjects={s.tickers} analysts={s.analysts} />
                      <span className="ml-auto text-slate-500">{nyTime(s.published_at)}</span>
                    </div>
                    <div className="mt-1">{s.text}</div>
                    {s.obligors && (
                      <div className="mt-1 text-xs text-rose-700 dark:text-rose-400">
                        → Module B: <span className="font-mono">{s.obligors.join(' ')}</span> · {s.gate}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
          <section className={card}>
            <h2 className={head}>Module A weights {r.weights ? `· set at close ${r.weights.date}` : ''}</h2>
            <div className="p-4">
              {r.weights ? (
                <WeightBars weights={r.weights.weights} />
              ) : (
                <p className="text-sm text-slate-500">Arrive once a trading day's news is in.</p>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
