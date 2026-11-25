import { useMemo, useState } from 'react'

function format(value, multiplier = 1) {
  return value == null ? '—' : (value * multiplier).toFixed(multiplier === 100 ? 1 : 3) + (multiplier === 100 ? '%' : '')
}

export default function Leaderboard({ metrics }) {
  const [sort, setSort] = useState({ key: 'accuracy', direction: 'desc' })
  const rows = useMemo(() => {
    const entries = Object.entries(metrics?.per_model || {}).map(([id, model]) => ({
      id,
      name: model.display_name,
      accuracy: model.raw?.selective_accuracy,
      ece: model.raw?.ece,
      brier: model.raw?.brier_score,
      coverage: model.raw?.coverage,
    }))
    return entries.sort((left, right) => {
      const a = left[sort.key]
      const b = right[sort.key]
      if (a == null || b == null) return a == null ? 1 : -1
      return (a - b) * (sort.direction === 'asc' ? 1 : -1)
    })
  }, [metrics, sort])

  function changeSort(key) {
    setSort(current => ({
      key,
      direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc',
    }))
  }

  if (!metrics) return <div className="alert alert-info">Complete the ARC evaluation to populate the model leaderboard.</div>
  const heading = (key, label) => (
    <button className="sort-button" type="button" onClick={() => changeSort(key)}>
      {label}{sort.key === key ? (sort.direction === 'desc' ? ' ↓' : ' ↑') : ''}
    </button>
  )
  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Model Leaderboard</h1>
        <p className="page-desc">
          Per-model test results from ARC-Challenge ({metrics.evaluation?.test_set_size ?? '—'} qualified test questions).
          Confidence metrics exclude responses without a valid self-reported confidence.
        </p>
      </header>
      <section className="card">
        <div className="table-wrapper">
          <table>
            <thead><tr>
              <th>Model</th>
              <th>{heading('accuracy', 'Selective accuracy')}</th>
              <th>{heading('ece', 'ECE')}</th>
              <th>{heading('brier', 'Brier score')}</th>
              <th>{heading('coverage', 'Coverage')}</th>
            </tr></thead>
            <tbody>{rows.map(model => {
              const details = metrics.per_model[model.id]
              return (
                <tr key={model.id}>
                  <td>{model.name}</td>
                  <td className="td-mono">{format(details.raw?.selective_accuracy, 100)}</td>
                  <td className="td-mono">{format(details.raw?.ece)}</td>
                  <td className="td-mono">{format(details.raw?.brier_score)}</td>
                  <td className="td-mono">{format(details.raw?.coverage, 100)}</td>
                </tr>
              )
            })}</tbody>
          </table>
        </div>
        <p className="form-hint" style={{ marginTop: 12 }}>Rows are sorted by {sort.key}, {sort.direction === 'desc' ? 'descending' : 'ascending'}.</p>
      </section>
    </div>
  )
}
