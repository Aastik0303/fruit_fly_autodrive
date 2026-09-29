import { PUSH_LEFT, PUSH_RIGHT } from '../../three/BrainActivity'

export default function DriversList({ drivers }) {
  return (
    <section className="card">
      <h3>Descending neurons tipping the choice</h3>
      {!drivers ? (
        <p className="hint">Appears once the game runs.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Cell type</th>
              <th>Side</th>
              <th className="num">Push</th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((driver) => (
              <tr key={driver.neuron_id} title={driver.neuron_id}>
                <td>{driver.cell_type}</td>
                <td>{driver.side}</td>
                <td className="num" style={{ color: driver.push > 0 ? PUSH_LEFT : PUSH_RIGHT }}>
                  {driver.push > 0 ? '← ' : '→ '}
                  {Math.abs(driver.push).toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="hint">
        Push = learned readout weight × current activity. The top 5 are marked in the brain view.
        The weights come from reinforcement learning, not from biology.
      </p>
    </section>
  )
}
