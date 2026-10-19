import { useRef } from 'react'
import { gsap, MOTION_OK, useGSAP } from '../motion'

interface Props {
  label: string
  value: string
  note?: string
  tone?: 'default' | 'alert'
  /** A key result: larger figure in the accent colour, for the top of a page. */
  hero?: boolean
}

// "-12.5%", "$1,204", "3 of 15": a leading text, one number, the rest. Anything else is left alone.
const NUMBER = /^([^\d-]*)(-?\d[\d,]*(?:\.\d+)?)(.*)$/

function parts(value: string) {
  const m = NUMBER.exec(value)
  if (!m) return null
  const digits = m[2].replace(/,/g, '')
  const decimals = digits.includes('.') ? digits.split('.')[1].length : 0
  return { prefix: m[1], number: Number(digits), decimals, comma: m[2].includes(','), suffix: m[3] }
}

export default function StatCard({ label, value, note, tone = 'default', hero = false }: Props) {
  const color =
    tone === 'alert'
      ? 'text-rose-600 dark:text-rose-400'
      : hero
        ? 'text-teal-700 dark:text-teal-300'
        : 'text-slate-900 dark:text-white'
  // A "before → after" figure is long: a step smaller so it stays on one line.
  const long = value.length > 13
  const size = hero
    ? long
      ? 'text-3xl sm:text-4xl'
      : 'text-4xl sm:text-5xl'
    : long
      ? 'text-2xl sm:text-3xl'
      : 'text-3xl sm:text-4xl'
  const figure = useRef<HTMLDivElement>(null)
  const shown = useRef<number | null>(null) // the number last drawn, so a change counts from it

  // Count to the new number when the value arrives or changes; the text always ends as `value`.
  useGSAP(
    () => {
      const el = figure.current
      const p = parts(value)
      if (!el || !p) {
        shown.current = null
        return
      }
      const from = shown.current ?? 0
      shown.current = p.number
      if (from === p.number) return
      const mm = gsap.matchMedia()
      mm.add(MOTION_OK, () => {
        const counter = { n: from }
        gsap.to(counter, {
          n: p.number,
          duration: 0.9,
          ease: 'power2.out',
          onUpdate: () => {
            const text = counter.n.toFixed(p.decimals)
            el.textContent =
              p.prefix +
              (p.comma ? Number(text).toLocaleString('en-US', { minimumFractionDigits: p.decimals }) : text) +
              p.suffix
          },
          onComplete: () => {
            el.textContent = value
          },
        })
      })
      return () => {
        mm.revert()
        el.textContent = value
      }
    },
    { dependencies: [value] },
  )

  return (
    <div
      className={`min-w-0 rounded-xl border bg-white dark:bg-slate-900 ${
        hero ? 'border-teal-200 p-5 dark:border-teal-900 sm:p-6' : 'border-slate-200 p-3 dark:border-slate-800 sm:p-4'
      }`}
    >
      <div className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</div>
      <div
        ref={figure}
        className={`mt-1 break-words font-figures font-semibold tabular-nums sm:whitespace-nowrap ${size} ${color}`}
      >
        {value}
      </div>
      {note && <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{note}</div>}
    </div>
  )
}
