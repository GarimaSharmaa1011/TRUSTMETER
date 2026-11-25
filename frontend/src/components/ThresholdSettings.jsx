import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'

function percentage(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

export default function ThresholdSettings({ metrics, threshold, setThreshold }) {
  const points = metrics?.risk_coverage?.trustmeter || []
  const selected = points.reduce((closest, point) => (
    !closest || Math.abs(point.threshold - threshold) < Math.abs(closest.threshold - threshold)
      ? point : closest
  ), null)
  const chartData = points.map(point => ({
    threshold: point.threshold,
    coverage: point.coverage * 100,
    risk: point.risk == null ? null : point.risk * 100,
  }))

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Threshold Settings</h1>
        <p className="page-desc">
          Explore saved ARC test-set coverage and risk for {metrics?.evaluation?.test_set_size ?? '—'} qualified questions
          at different calibrated trust thresholds.
        </p>
      </header>
      <div className="alert alert-info">
        This slider updates the saved test-set estimate only; it does not retrain the trust estimator. A higher threshold
        generally means less coverage. The live verdict uses the same selected threshold.
      </div>
      <div className="grid-2">
        <section className="card">
          <div className="card-header"><h2 className="card-title">Choose minimum trust</h2></div>
          <div className="slider-container">
            <div className="slider-header">
              <label className="slider-label" htmlFor="trust-threshold">Threshold</label>
              <span className="slider-value">{threshold.toFixed(2)}</span>
            </div>
            <input id="trust-threshold" type="range" min="0" max="1" step="0.01" value={threshold}
              onChange={event => setThreshold(Number(event.target.value))} />
            <div className="slider-range-labels"><span>0.00 · more answers</span><span>1.00 · stricter</span></div>
          </div>
          {selected
            ? <div className="metric-cards">
              <div className="metric-card"><div className="metric-card-value">{percentage(selected.coverage)}</div><div className="metric-card-label">Test coverage</div></div>
              <div className="metric-card"><div className="metric-card-value">{percentage(selected.risk == null ? null : 1 - selected.risk)}</div><div className="metric-card-label">Answered accuracy</div></div>
              <div className="metric-card"><div className="metric-card-value">{percentage(1 - selected.coverage)}</div><div className="metric-card-label">Test abstention rate</div></div>
            </div>
            : <div className="alert alert-info">No saved test curve is available. Run a complete ARC evaluation first.</div>}
          <p className="form-hint">Live verdicts remain estimates and should not be treated as proof.</p>
        </section>
        <section className="card">
          <div className="card-header"><h2 className="card-title">Test risk versus coverage</h2></div>
          <div className="chart-container" style={{ height: 340 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8eef5" />
                <XAxis dataKey="coverage" type="number" domain={[0, 100]} unit="%" />
                <YAxis type="number" domain={[0, 100]} unit="%" />
                <Tooltip labelFormatter={value => `Coverage ${Number(value).toFixed(1)}%`}
                  formatter={value => value == null ? 'No answered questions' : `${Number(value).toFixed(1)}%`} />
                {selected && <ReferenceLine x={selected.coverage * 100} stroke="#dc2626" strokeDasharray="4 4" />}
                <Line dataKey="risk" name="Error rate among answered" stroke="#dc2626" dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </div>
  )
}
