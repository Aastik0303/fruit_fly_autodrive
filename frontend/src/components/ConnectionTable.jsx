import { ROLE_COLORS } from '../colors'

const SIGN_LABEL = { 1: '+', '-1': '−', 0: '0' }

export default function ConnectionTable({ title, role, data, minSynapses, onSelect }) {
  if (!data) return null
  return (
    <section className="connections">
      <h3>
        <span className="dot" style={{ background: ROLE_COLORS[role] }} />
        {title}
        <small>{data.shown} of {data.total.toLocaleString()} (≥ {minSynapses} syn)</small>
      </h3>
      {data.items.length === 0 ? (
        <p className="hint">No connections at this threshold.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Cell type</th>
              <th>Class</th>
              <th className="num" title="number of synapses">Syn</th>
              <th className="num" title="share of the receiving neuron's total input">Weight</th>
              <th title="+ excitatory, − inhibitory, 0 unknown">±</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((item) => (
              <tr key={item.neuron_id} onClick={() => onSelect(item.neuron_id)} title={`Open ${item.neuron_id}`}>
                <td>{item.cell_type}</td>
                <td>{item.super_class}</td>
                <td className="num">{item.synapse_count}</td>
                <td className="num">{(item.weight * 100).toFixed(1)}%</td>
                <td className={`sign sign-${item.sign}`}>{SIGN_LABEL[item.sign]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
