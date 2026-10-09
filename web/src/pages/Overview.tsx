import { useApi } from '../api'
import ErrorBox from '../components/ErrorBox'
import EventMixChart from '../components/EventMixChart'
import ScoreTable from '../components/ScoreTable'
import StatCard from '../components/StatCard'
import type { Overview as OverviewData, Scoreboard } from '../types'

export default function Overview() {
  const overview = useApi<OverviewData>('/overview')
  const scores = useApi<Scoreboard>('/scoreboard')
  const error = overview.error ?? scores.error
  if (error) return <ErrorBox message={error} />
  if (!overview.data || !scores.data) return <p className="text-sm text-slate-500">Loading…</p>

  const o = overview.data
  const [sentimentTask, eventsTask] = Object.keys(scores.data)
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Texts scored" value={o.texts.toLocaleString()} note={`${o.signals.toLocaleString()} text–stock signals`} />
        <StatCard
          label="News / tweets"
          value={`${(o.texts_by_source.news ?? 0).toLocaleString()} / ${(o.texts_by_source.tweets ?? 0).toLocaleString()}`}
          note="FNSPID headlines · 2020 stock tweets"
        />
        <StatCard
          label={`High impact (> ${o.config.impact_trigger})`}
          value={o.alerts.toLocaleString()}
          note="impact on a 1–10 scale"
          tone="alert"
        />
        <StatCard label="Trading days" value={String(o.trading_days)} note={`${o.config.window_start} → ${o.config.window_end}`} />
      </div>

      <EventMixChart events={o.events} />

      <div className="grid gap-6 xl:grid-cols-2">
        <ScoreTable title={sentimentTask} rows={scores.data[sentimentTask]} engineModel={o.engine_models.sentiment} />
        <ScoreTable title={eventsTask} rows={scores.data[eventsTask]} engineModel={o.engine_models.events} />
      </div>
    </div>
  )
}
