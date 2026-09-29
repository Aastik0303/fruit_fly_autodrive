import { CLASS_COLORS, ROLE_COLORS } from '../colors'

export default function Legend({ classes, counts }) {
  if (!classes) return null
  return (
    <section className="legend">
      <h3>Legend</h3>
      <ul>
        {classes.map((name) => (
          <li key={name}>
            <span className="dot" style={{ background: CLASS_COLORS[name] ?? CLASS_COLORS.unknown }} />
            {name.replaceAll('_', ' ')}
            <small>{counts?.[name]?.toLocaleString()}</small>
          </li>
        ))}
        <li className="legend-gap"><span className="dot" style={{ background: ROLE_COLORS.selected }} />selected neuron</li>
        <li><span className="dot" style={{ background: ROLE_COLORS.incoming }} />incoming partner</li>
        <li><span className="dot" style={{ background: ROLE_COLORS.outgoing }} />outgoing partner</li>
      </ul>
    </section>
  )
}
