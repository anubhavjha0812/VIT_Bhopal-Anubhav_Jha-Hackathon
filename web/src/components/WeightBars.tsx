import { useRef } from 'react'
import { gsap, MOTION_OK, useGSAP } from '../motion'

const CAP = 0.15 // Module A's per-stock cap: a full bar

/** One bar per stock, largest first; the tick marks equal weight. The bars grow out of the
 *  left edge once when the allocation first appears. */
export default function WeightBars({ weights }: { weights: Record<string, number | string> }) {
  const list = useRef<HTMLUListElement>(null)
  const entries = Object.entries(weights)
    .filter(([k]) => k !== 'date')
    .map(([k, v]) => [k, Number(v)] as const)
    .sort((a, b) => b[1] - a[1])
  const equal = 1 / entries.length

  useGSAP(
    () => {
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        gsap.from('[data-bar]', {
          scaleX: 0,
          transformOrigin: '0 50%',
          duration: 0.8,
          ease: 'power3.out',
          stagger: 0.03,
          clearProps: 'transform',
        })
      })
    },
    { scope: list },
  )

  return (
    <ul ref={list} className="space-y-1.5">
      {entries.map(([t, w]) => (
        <li key={t} className="flex items-center gap-2 text-sm">
          <span className="w-14 font-figures">{t}</span>
          <div className="relative h-3 flex-1 rounded bg-slate-100 dark:bg-slate-800">
            <div
              data-bar
              className="absolute h-3 rounded bg-teal-500 transition-[width] duration-500"
              style={{ width: `${(w / CAP) * 100}%` }}
            />
            <div className="absolute top-[-2px] h-4 w-px bg-slate-500" style={{ left: `${(equal / CAP) * 100}%` }} />
          </div>
          <span className="w-12 text-right font-figures tabular-nums">{(w * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  )
}
