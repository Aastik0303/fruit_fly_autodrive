export default function ReportTable({ report }) {
  if (!report) return null
  return (
    <section className="card">
      <h3>Evaluation · {report.episodes} games of 100 s</h3>
      <table>
        <thead>
          <tr>
            <th>Policy</th>
            <th className="num">Score</th>
            <th className="num">Hits/min</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(report.results).map(([name, result]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="num">{result.mean_score}</td>
              <td className="num">{result.hits_per_minute}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">{report.note}</p>
    </section>
  )
}
