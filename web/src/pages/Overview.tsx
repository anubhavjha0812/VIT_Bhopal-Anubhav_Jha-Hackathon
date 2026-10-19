import { useApi } from '../api'
import ErrorBox from '../components/ErrorBox'
import EventMixChart from '../components/EventMixChart'
import ClassF1Table from '../components/ClassF1Table'
import ScoreTable from '../components/ScoreTable'
import HeadlineScores from '../components/HeadlineScores'
import StatCard from '../components/StatCard'
import type { Overview as OverviewData, Scoreboard } from '../types'

export default function Overview() {
  const overview = useApi<OverviewData>('/overview')
  const scores = useApi<Scoreboard>('/scoreboard')
  const error = overview.error ?? scores.error
  if (error) return <ErrorBox message={error} />
  if (!overview.data || !scores.data) return <p className="text-sm text-slate-500">Loading…</p>

  const o = overview.data
  const [sentimentTask, eventsTask, events9Task] = Object.keys(scores.data)
  // The scoreboard row that matches what runs. Events: the 9-class int8 model (plus rules). Sentiment:
  // the company-specific model has no row on this tweet test, so its fallback row is marked.
  const liveEvents = o.engine_models.events.startsWith('keywords') ? 'keywords (baseline)' : 'events9-bge-small-en-v1.5-int8'
  const liveSentiment = 'sentiment-bge-small-en-v1.5-int8'
  const sentimentNote =
    o.config.sentiment === 'entity'
      ? 'Used when a text names no company. Company-specific sentiment runs a separate model, scored on SEntFiN (see the results above).'
      : 'The model the running engine uses'
  return (
    <div className="space-y-10">
      <HeadlineScores />
      <h2 className="text-2xl font-semibold text-slate-900 dark:text-white">What was run</h2>
      <div className="grid gap-3 sm:grid-cols-2 sm:gap-4 lg:grid-cols-4">
        <StatCard label="Texts scored" value={o.texts.toLocaleString()} note={`${o.signals.toLocaleString()} text–stock signals`} />
        <StatCard
          label="News / tweets"
          value={`${(o.texts_by_source.news ?? 0).toLocaleString()} / ${(o.texts_by_source.tweets ?? 0).toLocaleString()}`}
          note="FNSPID headlines · 2020 stock tweets"
        />
        <StatCard
          label="Credit alerts → Module B"
          value={o.alerts.toLocaleString()}
          note={`Credit gate passed ${o.alerts} of ${o.alerts_naive} texts with impact > ${o.config.impact_trigger}`}
          tone="alert"
        />
        <StatCard label="Trading days" value={String(o.trading_days)} note={`${o.config.window_start} → ${o.config.window_end}`} />
      </div>

      <EventMixChart events={o.events} />

      <div className="grid min-w-0 gap-6 xl:grid-cols-2">
        <ScoreTable title={sentimentTask} rows={scores.data[sentimentTask]} engineModel={liveSentiment} liveNote={sentimentNote} />
        <ScoreTable title={eventsTask} rows={scores.data[eventsTask]} engineModel={liveEvents} />
      </div>

      {events9Task && (
        <div className="grid min-w-0 gap-6 xl:grid-cols-2">
          <ScoreTable title={events9Task} rows={scores.data[events9Task]} engineModel={liveEvents} liveNote="The event model the engine runs, with the rules on top (E24 candidate C)" />
          <ClassF1Table rows={scores.data[events9Task]} engineModel={liveEvents} />
        </div>
      )}
    </div>
  )
}
