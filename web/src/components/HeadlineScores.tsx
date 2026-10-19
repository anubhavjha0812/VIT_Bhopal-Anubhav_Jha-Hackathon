import { useRef } from 'react'
import { HEADLINES, TAGS, type Headline } from '../headline'
import { gsap, MOTION_OK, useGSAP } from '../motion'

const TAG_STYLE = {
  human: 'bg-teal-50 text-teal-800 dark:bg-teal-950 dark:text-teal-200',
  design: 'bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200',
  ai: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200',
} as const

function Tile({ h, primary }: { h: Headline; primary: boolean }) {
  const root = useRef<HTMLElement>(null)
  const figure = useRef<HTMLDivElement>(null)
  const top = Math.max(h.value, h.baseline)
  const pct = (n: number) => `${(n / top) * 100}%`

  // The number counts up and the two bars grow out of the left edge, once, when the tile appears.
  useGSAP(
    () => {
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        const counter = { n: 0 }
        gsap.to(counter, {
          n: h.value,
          duration: 1.1,
          ease: 'power3.out',
          onUpdate: () => {
            if (figure.current) figure.current.textContent = counter.n.toFixed(h.decimals)
          },
          onComplete: () => {
            if (figure.current) figure.current.textContent = h.value.toFixed(h.decimals)
          },
        })
        gsap.from('[data-bar]', { scaleX: 0, transformOrigin: '0 50%', duration: 0.9, ease: 'power3.out', stagger: 0.12, clearProps: 'transform' })
      })
    },
    { scope: root },
  )

  return (
    <article
      ref={root}
      className={`flex min-w-0 flex-col gap-4 rounded-[20px] border bg-white p-6 ${primary ? 'border-teal-300 dark:border-teal-700' : 'border-slate-200 dark:border-slate-800'} shadow-[0_1px_2px_rgba(23,34,56,0.06),0_16px_36px_rgba(23,34,56,0.07)] dark:bg-slate-900 dark:shadow-none`}
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="min-h-[2.5rem] text-sm font-medium text-slate-600 dark:text-slate-300">{h.label}</h3>
        <span title={TAGS[h.tag].hint} className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium ${TAG_STYLE[h.tag]}`}>
          {TAGS[h.tag].text}
        </span>
      </div>
      <div
        ref={figure}
        className="font-figures text-7xl font-semibold leading-none tabular-nums tracking-tight text-teal-700 dark:text-teal-300"
      >
        {h.value.toFixed(h.decimals)}
      </div>
      <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">{h.note}</p>
      <div className="mt-auto space-y-1.5" aria-label={`${h.value} against ${h.baseline} for ${h.baselineName}`}>
        <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
          <div className="h-2 flex-1 rounded-full bg-slate-100 dark:bg-slate-800">
            <div data-bar className="h-2 rounded-full bg-slate-400" style={{ width: pct(h.baseline) }} />
          </div>
          <span className="w-36 shrink-0 tabular-nums">
            {h.baseline.toFixed(h.decimals)} {h.baselineName}
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs font-medium text-slate-800 dark:text-slate-100">
          <div className="h-2 flex-1 rounded-full bg-slate-100 dark:bg-slate-800">
            <div data-bar className="h-2 rounded-full bg-teal-500" style={{ width: pct(h.value) }} />
          </div>
          <span className="w-36 shrink-0 tabular-nums">{h.value.toFixed(h.decimals)} Helix</span>
        </div>
      </div>
      <p className="font-mono text-[11px] text-slate-400 dark:text-slate-500">{h.source}</p>
    </article>
  )
}

/** The first thing on the Overview: four checked numbers, each against its baseline. */
export default function HeadlineScores() {
  return (
    <section className="space-y-5">
      <div className="max-w-5xl">
        <h1 className="text-5xl font-semibold text-slate-900 dark:text-white sm:text-6xl">From financial text to measurable risk</h1>
        <p className="mt-3 text-base leading-relaxed text-slate-600 dark:text-slate-300">
          Helix reads a headline once, then carries it to a portfolio weight and a bank capital number. These are the four
          results to read first. Each is shown against its baseline, with how far to trust it.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {HEADLINES.map((h, i) => (
          <Tile key={h.label} h={h} primary={i === 0} />
        ))}
      </div>
      <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Design data: the credit-gate rules were shaped on these rows, so it is not an unseen test. AI labels: ChatGPT and Claude, with
        no human check. Module A shows no proven edge over equal weight; see the Module A page. Full results: docs/others/RESULTS.md.
      </p>
    </section>
  )
}
