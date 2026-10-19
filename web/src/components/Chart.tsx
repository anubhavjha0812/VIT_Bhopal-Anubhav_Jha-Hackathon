// ECharts with only the parts we use (the full library is ~1 MB of JS).
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, MarkLineComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import type { CSSProperties } from 'react'

echarts.use([
  BarChart, LineChart, GridComponent, LegendComponent, MarkLineComponent, TooltipComponent, CanvasRenderer,
])

interface Props {
  option: object
  style?: CSSProperties
}

// Canvas text does not inherit the page font, so name it here (a chart may still set its own).
const TEXT = { textStyle: { fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif" } }

export default function Chart({ option, style }: Props) {
  return <ReactEChartsCore echarts={echarts} option={{ ...TEXT, ...option }} style={{ width: '100%', ...style }} notMerge />
}
