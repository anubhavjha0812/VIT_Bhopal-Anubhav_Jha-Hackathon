// ECharts with only the parts we use (the full library is ~1 MB of JS).
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import type { CSSProperties } from 'react'

echarts.use([BarChart, LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

interface Props {
  option: object
  style?: CSSProperties
}

export default function Chart({ option, style }: Props) {
  return <ReactEChartsCore echarts={echarts} option={option} style={style} notMerge />
}
