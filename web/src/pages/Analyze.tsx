import { useState, type FormEvent } from 'react'
import { signed } from '../format'
import { postJSON } from '../api'
import ErrorBox from '../components/ErrorBox'
import EventBadge, { ImpactPill } from '../components/EventBadge'
import Tickers from '../components/Tickers'
import type { Analysis } from '../types'

const EXAMPLES = [
  'JPMorgan warns of rising loan defaults as unemployment surges',
  'Microsoft to acquire a cybersecurity startup for $1.2 billion',
  'Fed cuts rates to near zero and launches $700 billion stimulus',
  'Tesla deliveries beat estimates; shares jump 8% after hours',
]

interface Result extends Analysis {
  text: string
  ms: number
}

function SentimentBar({ value }: { value: number }) {
  // -1 … +1 drawn from the centre line
  const width = `${Math.abs(value) * 50}%`
  const left = value >= 0 ? '50%' : `${50 - Math.abs(value) * 50}%`
  return (
    <div className="relative h-2 rounded bg-slate-100 dark:bg-slate-800">
      <div className="absolute top-0 h-2 w-px bg-slate-400" style={{ left: '50%' }} />
      <div
        className={`absolute top-0 h-2 rounded ${value >= 0 ? 'bg-teal-500' : 'bg-rose-500'}`}
        style={{ left, width }}
      />
    </div>
  )
}

export default function Analyze() {
  const [text, setText] = useState(EXAMPLES[0])
  const [source, setSource] = useState<'news' | 'tweets'>('news')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<Result[]>([])

  async function run(e: FormEvent) {
    e.preventDefault()
    if (!text.trim()) return
    setBusy(true)
    setError(null)
    const t0 = performance.now()
    try {
      const out = await postJSON<Analysis>('/analyze', { text, source })
      setHistory((h) => [{ ...out, text, ms: performance.now() - t0 }, ...h].slice(0, 10))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const latest = history[0]
  return (
    <div className="grid gap-6 xl:grid-cols-2">
      <form onSubmit={run} className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
        <h2 className="text-sm font-semibold">Score any headline or post live</h2>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          maxLength={1000}
          className="w-full rounded-lg border border-slate-200 bg-white p-3 text-sm dark:border-slate-700 dark:bg-slate-950"
          placeholder="e.g. Boeing suspends dividend as airlines cancel orders"
        />
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={source}
            onChange={(e) => setSource(e.target.value as 'news' | 'tweets')}
            className="rounded-lg border border-slate-200 bg-white px-2 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
          >
            <option value="news">News headline</option>
            <option value="tweets">Social post</option>
          </select>
          <button
            type="submit"
            disabled={busy}
            className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-700 disabled:opacity-60"
          >
            {busy ? 'Scoring…' : 'Analyze'}
          </button>
        </div>
        <div className="space-y-1">
          <div className="text-xs text-slate-500">Try:</div>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setText(ex)}
              className="block text-left text-sm text-teal-700 hover:underline dark:text-teal-400"
            >
              {ex}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Runs our int8 models on the server CPU. Companies are found from the 15 stocks' names and cashtags. The first
          request after a restart can take a few seconds while the models load.
        </p>
      </form>

      <div className="space-y-4">
        {error && <ErrorBox message={error} />}
        {latest && (
          <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <p className="text-sm">“{latest.text}”</p>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-xs uppercase tracking-wide text-slate-500">Sentiment</div>
                <div className="font-mono text-2xl font-semibold tabular-nums">
                  {signed(latest.sentiment)}
                </div>
                <SentimentBar value={latest.sentiment} />
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-slate-500">Event</div>
                <div className="mt-2">
                  <EventBadge event={latest.event} />
                  {latest.analyst_rating && (
                    <span className="ml-2 text-xs text-slate-500 dark:text-slate-400">analyst rating, not credit</span>
                  )}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-slate-500">Impact (1–10)</div>
                <div className="mt-2">
                  <ImpactPill impact={latest.impact} />
                </div>
              </div>
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              Companies:{' '}
              {latest.tickers.length || latest.analysts.length ? (
                <Tickers subjects={latest.tickers} analysts={latest.analysts} />
              ) : (
                'none of the 15 found'
              )}{' '}
              · model{' '}
              <span className="font-mono">{latest.model}</span> · {Math.round(latest.ms)} ms round trip
            </div>
          </section>
        )}
        {history.length > 1 && (
          <section className="rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
            <h2 className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">Earlier</h2>
            <ul>
              {history.slice(1).map((r, i) => (
                <li key={i} className="flex items-center gap-3 border-t border-slate-100 px-4 py-2 text-sm first:border-t-0 dark:border-slate-800">
                  <span className="w-14 text-right font-mono tabular-nums">
                    {signed(r.sentiment)}
                  </span>
                  <EventBadge event={r.event} />
                  <ImpactPill impact={r.impact} />
                  <span className="truncate">{r.text}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}
