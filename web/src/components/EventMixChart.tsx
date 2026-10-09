import { chartColors, useTheme } from '../theme'
import Chart from './Chart'

interface Props {
  events: Record<string, number>
}

// "Other" dwarfs every class (93% of texts), so it is shown as a note, not a bar.
export default function EventMixChart({ events }: Props) {
  const theme = useTheme()
  const rows = Object.entries(events)
    .filter(([name]) => name !== 'Other')
    .sort((a, b) => a[1] - b[1])
  const total = Object.values(events).reduce((a, b) => a + b, 0)
  const other = events.Other ?? 0
  const { text, grid } = chartColors(theme)

  const option = {
    backgroundColor: 'transparent',
    grid: { left: 130, right: 40, top: 10, bottom: 30 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: {
      type: 'value',
      axisLabel: { color: text },
      splitLine: { lineStyle: { color: grid } },
    },
    yAxis: {
      type: 'category',
      data: rows.map(([name]) => name),
      axisLabel: { color: text },
      axisLine: { lineStyle: { color: grid } },
    },
    series: [
      {
        type: 'bar',
        data: rows.map(([name, n]) => ({
          value: n,
          itemStyle: {
            color: ['Default', 'Credit downgrade', 'Covenant breach'].includes(name) ? '#e11d48' : '#0d9488',
            borderRadius: [0, 4, 4, 0],
          },
        })),
        label: { show: true, position: 'right', color: text },
      },
    ],
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
      <h2 className="border-b border-slate-200 px-4 py-3 text-sm font-semibold dark:border-slate-800">
        Event classes in the window
      </h2>
      <Chart option={option} style={{ height: 260 }} />
      <p className="px-4 pb-3 text-xs text-slate-500 dark:text-slate-400">
        Red = credit events. Not shown: Other,{' '}
        {other.toLocaleString()} texts ({((other / total) * 100).toFixed(0)}%).
      </p>
    </section>
  )
}
