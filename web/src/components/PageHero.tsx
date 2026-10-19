import type { ReactNode } from 'react'

/** The top of a page: what it is in one line, a short plain description, then the key results. */
export default function PageHero({
  title,
  text,
  wide = false,
  children,
}: {
  title: string
  text?: ReactNode
  /** Let the description run the full page width instead of a readable column. */
  wide?: boolean
  children?: ReactNode
}) {
  return (
    <header className="space-y-4">
      <div className={`space-y-2 ${wide ? '' : 'max-w-4xl'}`}>
        <h1 className="text-4xl font-semibold text-slate-900 dark:text-white sm:text-5xl">{title}</h1>
        {text && <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300 sm:text-base">{text}</p>}
      </div>
      {children}
    </header>
  )
}
