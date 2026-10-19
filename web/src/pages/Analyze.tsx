import { useEffect, useRef, useState, type FormEvent } from 'react'
import { gsap, MOTION_OK, useGSAP } from '../motion'
import { signed } from '../format'
import { postJSON } from '../api'
import ErrorBox from '../components/ErrorBox'
import PageHero from '../components/PageHero'
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
    <div className="relative h-2.5 rounded-full bg-slate-100 dark:bg-slate-800">
      <div className="absolute -top-1 h-4.5 w-px bg-slate-400" style={{ left: '50%' }} />
      <div
        data-sent-fill
        className={`absolute top-0 h-2.5 rounded-full ${value >= 0 ? 'bg-teal-500' : 'bg-rose-500'}`}
        style={{ left, width }}
      />
    </div>
  )
}

const mood = (x: number) => (x > 0.15 ? 'Positive' : x < -0.15 ? 'Negative' : 'Neutral')

const tile = 'min-w-0 rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900'
const label = 'text-xs font-medium text-slate-500 dark:text-slate-400'

export default function Analyze() {
  const [text, setText] = useState(EXAMPLES[0])
  const [source, setSource] = useState<'news' | 'tweets'>('news')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<Result[]>([])

  async function analyze(input: string, from: 'news' | 'tweets') {
    if (!input.trim()) return
    setBusy(true)
    setError(null)
    const t0 = performance.now()
    try {
      const out = await postJSON<Analysis>('/analyze', { text: input, source: from })
      setHistory((h) => [{ ...out, text: input, ms: performance.now() - t0 }, ...h].slice(0, 10))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  function run(e: FormEvent) {
    e.preventDefault()
    void analyze(text, source)
  }

  // Score the first example on arrival, so the page opens with a real answer instead of empty space.
  const started = useRef(false)
  useEffect(() => {
    if (started.current) return
    started.current = true
    void analyze(EXAMPLES[0], 'news')
  }, [])

  const latest = history[0]
  const result = useRef<HTMLElement>(null)

  // A new answer: the panel settles in and the sentiment bar draws out from the centre line.
  useGSAP(
    () => {
      if (!latest) return
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        gsap
          .timeline({ defaults: { ease: 'power3.out' } })
          .from(result.current, { opacity: 0, y: 10, duration: 0.35, clearProps: 'all' })
          .from(
            '[data-sent-fill]',
            {
              scaleX: 0,
              transformOrigin: latest.sentiment >= 0 ? '0 50%' : '100% 50%',
              duration: 0.6,
              clearProps: 'transform',
            },
            '-=0.15',
          )
          .from('[data-pop]', { scale: 0.85, opacity: 0, duration: 0.35, stagger: 0.08, clearProps: 'all' }, '-=0.45')
      })
    },
    { scope: result, dependencies: [latest] },
  )

  const segment = (on: boolean) =>
    `rounded-md px-3 py-1.5 text-sm transition ${
      on
        ? 'bg-white font-semibold text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white'
        : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white'
    }`

  return (
    <div className="space-y-6">
      <PageHero
        wide
        title="Score any headline"
        text="Type or paste a headline or post. Helix finds the company, reads sentiment, names the event and rates the impact on a 1 to 10 scale. It runs on the server CPU."
      />

      <form
        onSubmit={run}
        className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900"
      >
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          maxLength={1000}
          className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-4 text-lg leading-relaxed outline-none transition focus:border-teal-500 focus:bg-white dark:border-slate-700 dark:bg-slate-950 dark:focus:border-teal-400"
          placeholder="e.g. Boeing suspends dividend as airlines cancel orders"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <div className="inline-flex rounded-lg bg-slate-100 p-1 dark:bg-slate-800" role="group" aria-label="Source">
            <button type="button" onClick={() => setSource('news')} className={segment(source === 'news')}>
              News headline
            </button>
            <button type="button" onClick={() => setSource('tweets')} className={segment(source === 'tweets')}>
              Social post
            </button>
          </div>
          <span className="text-xs text-slate-500 dark:text-slate-400">{text.length} / 1000</span>
          <button
            type="submit"
            disabled={busy || !text.trim()}
            className="ml-auto w-full rounded-lg bg-teal-600 px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-teal-700 disabled:opacity-60 sm:w-auto"
          >
            {busy ? 'Scoring…' : 'Analyze'}
          </button>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-500 dark:text-slate-400">Try:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setText(ex)
                void analyze(ex, source)
              }}
              className="max-w-full truncate rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-700 transition hover:border-teal-500 hover:text-teal-700 dark:border-slate-700 dark:text-slate-300 dark:hover:border-teal-400 dark:hover:text-teal-300"
            >
              {ex}
            </button>
          ))}
        </div>
      </form>

      {error && <ErrorBox message={error} />}

      {latest && (
        <section ref={result} className="space-y-4" aria-busy={busy}>
          <p className="rounded-xl border-l-4 border-teal-600 bg-white px-5 py-4 text-lg leading-relaxed dark:bg-slate-900">
            “{latest.text}”
          </p>
          <div className={`grid gap-4 sm:grid-cols-2 xl:grid-cols-4 transition ${busy ? 'opacity-50' : ''}`}>
            <div className={tile}>
              <div className={label}>Sentiment</div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-figures text-5xl font-semibold tabular-nums">{signed(latest.sentiment)}</span>
                <span
                  className={`text-sm font-medium ${
                    latest.sentiment > 0.15
                      ? 'text-teal-700 dark:text-teal-300'
                      : latest.sentiment < -0.15
                        ? 'text-rose-600 dark:text-rose-400'
                        : 'text-slate-500'
                  }`}
                >
                  {mood(latest.sentiment)}
                </span>
              </div>
              <div className="mt-4">
                <SentimentBar value={latest.sentiment} />
                <div className="mt-1.5 flex justify-between text-[11px] text-slate-400">
                  <span>−1</span>
                  <span>0</span>
                  <span>+1</span>
                </div>
              </div>
            </div>
            <div className={tile}>
              <div className={label}>Event</div>
              <div className="mt-4" data-pop>
                <div className="origin-left scale-150">
                  <EventBadge event={latest.event} />
                </div>
              </div>
              {latest.analyst_rating && (
                <div className="mt-3 text-xs text-slate-500 dark:text-slate-400">analyst rating, not credit</div>
              )}
            </div>
            <div className={tile}>
              <div className={label}>Impact (1–10)</div>
              <div className="mt-4" data-pop>
                <div className="origin-left scale-150">
                  <ImpactPill impact={latest.impact} />
                </div>
              </div>
              <div className="mt-3 text-xs text-slate-500 dark:text-slate-400">
                7 and above can trigger a credit alert
              </div>
            </div>
            <div className={tile}>
              <div className={label}>Companies</div>
              <div className="mt-3 text-sm">
                {latest.tickers.length || latest.analysts.length ? (
                  <Tickers subjects={latest.tickers} analysts={latest.analysts} />
                ) : (
                  <span className="text-slate-500 dark:text-slate-400">no known company found</span>
                )}
              </div>
            </div>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Model <span className="font-figures">{latest.model}</span> · {Math.round(latest.ms)} ms round trip.
            Companies are found from 99 names and cashtags; 15 of them are in the portfolio.
          </p>
        </section>
      )}

      {history.length > 1 && (
        <section className="rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
          <h2 className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">Earlier</h2>
          <ul>
            {history.slice(1).map((r, i) => (
              <li
                key={i}
                className="flex flex-wrap items-center gap-3 border-t border-slate-100 px-4 py-2.5 text-sm first:border-t-0 dark:border-slate-800 sm:flex-nowrap"
              >
                <span className="w-14 shrink-0 text-right font-figures tabular-nums">{signed(r.sentiment)}</span>
                <EventBadge event={r.event} />
                <ImpactPill impact={r.impact} />
                <span className="min-w-0 flex-1 truncate">{r.text}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
