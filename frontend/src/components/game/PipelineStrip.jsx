export default function PipelineStrip({ hello }) {
  const steps = [
    ['Game', 'falling blocks, 80 FPS'],
    ['Eyes', `${hello?.photoreceptors?.toLocaleString() ?? '…'} R7/R8 photoreceptors`],
    ['Connectome', `${hello?.hops ?? '…'} synaptic hops through FlyWire v783`],
    ['Descending neurons', `${hello?.descending?.toLocaleString() ?? '…'} brain → body neurons`],
    ['RL readout', 'Q(left), Q(stay), Q(right)'],
    ['Move', 'every 4 frames'],
  ]
  return (
    <section className="card pipeline">
      <h3>How the fly decides</h3>
      <ol>
        {steps.map(([title, detail]) => (
          <li key={title}>
            <strong>{title}</strong>
            <span>{detail}</span>
          </li>
        ))}
      </ol>
      <p className="hint">
        Brain activity is a linear signal-flow model on the real wiring (no spikes or timing), not a
        simulation of real fly neurons. Only the readout is learned; the connectome stays fixed.
      </p>
    </section>
  )
}
