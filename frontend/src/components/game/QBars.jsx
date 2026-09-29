const LABELS = ['left', 'stay', 'right']

export default function QBars({ decision }) {
  if (!decision) {
    return (
      <section className="card">
        <h3>Decision</h3>
        <p className="hint">Press Start to watch the fly decide 20 times per second.</p>
      </section>
    )
  }
  const { q, action, policy } = decision
  const low = Math.min(...q)
  const span = Math.max(...q) - low || 1

  return (
    <section className="card">
      <h3>Decision: Q-values</h3>
      {LABELS.map((label, i) => (
        <div key={label} className={`qbar ${label === action ? 'chosen' : ''}`}>
          <span>{label}</span>
          <div className="track">
            <div className="fill" style={{ width: `${8 + (92 * (q[i] - low)) / span}%` }} />
          </div>
          <code>{q[i].toFixed(2)}</code>
        </div>
      ))}
      <p className="hint">
        {policy === 'random'
          ? 'Random baseline: the move is picked at random; Q-values are shown for comparison only.'
          : 'Q = expected future reward of each move, read out from descending-neuron activity. The highest one is chosen.'}
      </p>
    </section>
  )
}
