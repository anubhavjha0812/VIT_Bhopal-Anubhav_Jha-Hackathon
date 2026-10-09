// Companies a text is about, then banks it names only as the stock-rating analyst (DECISIONS E12).
export function AnalystTag({ ticker }: { ticker: string }) {
  return (
    <span
      title="Named only as the analyst giving a stock rating; not what the text is about"
      className="whitespace-nowrap rounded bg-slate-100 px-1 font-mono text-xs text-slate-500 dark:bg-slate-800 dark:text-slate-400"
    >
      {ticker} (analyst)
    </span>
  )
}

export default function Tickers({ subjects, analysts = [] }: { subjects: string[]; analysts?: string[] }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-1 font-mono text-xs">
      {subjects.length > 0 && <span>{subjects.join(' ')}</span>}
      {analysts.map((t) => (
        <AnalystTag key={t} ticker={t} />
      ))}
    </span>
  )
}
