import ConnectionTable from './ConnectionTable'

const THRESHOLDS = [1, 2, 5, 10, 20]

export default function NeuronPanel({ neuron, connections, minSynapses, onMinSynapses, onSelect, onClear }) {
  const transmitter = neuron.nt_confidence == null
    ? neuron.neurotransmitter
    : `${neuron.neurotransmitter} (confidence ${neuron.nt_confidence.toFixed(2)})`

  const rows = [
    ['Neuron ID', neuron.neuron_id],
    ['Super class', neuron.super_class],
    ['Class', neuron.cell_class],
    ['Flow', neuron.flow],
    ['Side', neuron.side],
    ['Neurotransmitter', transmitter],
    ['Position (µm)', neuron.position_um ? neuron.position_um.map((v) => v.toFixed(0)).join(', ') : 'not available'],
    ['Inputs', `${neuron.in_degree.toLocaleString()} neurons · ${neuron.in_synapses.toLocaleString()} synapses`],
    ['Outputs', `${neuron.out_degree.toLocaleString()} neurons · ${neuron.out_synapses.toLocaleString()} synapses`],
  ]

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>{neuron.cell_type}</h2>
        <button type="button" onClick={onClear}>Clear</button>
      </div>
      <dl className="meta">
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {neuron.is_outlier && <p className="warning">Flagged as an outlier in the FlyWire annotations.</p>}
      <label className="threshold">
        Minimum synapses
        <select value={minSynapses} onChange={(event) => onMinSynapses(Number(event.target.value))}>
          {THRESHOLDS.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </label>
      <ConnectionTable title="Incoming" role="incoming" data={connections.incoming} minSynapses={minSynapses} onSelect={onSelect} />
      <ConnectionTable title="Outgoing" role="outgoing" data={connections.outgoing} minSynapses={minSynapses} onSelect={onSelect} />
    </div>
  )
}
