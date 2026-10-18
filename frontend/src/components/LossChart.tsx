import { ChartNoAxesCombined } from 'lucide-react'

interface LossChartProps {
  history: number[]
  label?: string
}

const WIDTH = 760
const HEIGHT = 250
const PADDING = { top: 18, right: 18, bottom: 34, left: 68 }

export function LossChart({ history, label = 'Training loss' }: LossChartProps) {
  const innerWidth = WIDTH - PADDING.left - PADDING.right
  const innerHeight = HEIGHT - PADDING.top - PADDING.bottom
  const min = history.length ? Math.min(...history) : 0
  const max = history.length ? Math.max(...history) : 1
  const range = max - min || Math.max(Math.abs(max) * 0.08, 1e-6)
  const points = history.map((value, index) => {
    const x = PADDING.left + (index / Math.max(history.length - 1, 1)) * innerWidth
    const y = PADDING.top + ((max - value) / range) * innerHeight
    return `${x},${y}`
  }).join(' ')

  return (
    <section className="surface-panel chart-panel" aria-labelledby="curve-title">
      <div className="chart-heading">
        <div>
          <div className="section-heading-row">
            <ChartNoAxesCombined size={19} strokeWidth={1.8} aria-hidden="true" />
            <h2 id="curve-title">Learning curve</h2>
          </div>
          <p className="section-copy">Training loss measured after each full pass through the data.</p>
        </div>
        {history.length > 0 && (
          <div className="chart-summary" aria-live="polite">
            <span>{label}</span>
            <strong>{history.at(-1)?.toFixed(4)}</strong>
          </div>
        )}
      </div>

      {history.length === 0 ? (
        <div className="chart-empty">
          <div className="chart-empty-icon"><ChartNoAxesCombined size={23} aria-hidden="true" /></div>
          <p>Your curve will appear here</p>
          <span>Train the model to see how the selected loss changes over time.</span>
        </div>
      ) : (
        <div className="chart-wrap">
          <svg className="loss-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${label} across ${history.length} training epochs`}>
            {[0, 0.5, 1].map((fraction) => {
              const y = PADDING.top + fraction * innerHeight
              const value = max - fraction * range
              return (
                <g key={fraction}>
                  <line x1={PADDING.left} x2={WIDTH - PADDING.right} y1={y} y2={y} className="chart-gridline" />
                  <text x={PADDING.left - 12} y={y + 4} textAnchor="end" className="chart-tick">{value.toFixed(3)}</text>
                </g>
              )
            })}
            <text x={PADDING.left} y={HEIGHT - 8} className="chart-tick">Epoch 1</text>
            <text x={WIDTH - PADDING.right} y={HEIGHT - 8} textAnchor="end" className="chart-tick">Epoch {history.length}</text>
            <polyline points={points} className="chart-line" />
            <circle
              cx={PADDING.left + ((history.length - 1) / Math.max(history.length - 1, 1)) * innerWidth}
              cy={PADDING.top + ((max - (history.at(-1) ?? 0)) / range) * innerHeight}
              r="4.5"
              className="chart-endpoint"
            />
          </svg>
        </div>
      )}
    </section>
  )
}
