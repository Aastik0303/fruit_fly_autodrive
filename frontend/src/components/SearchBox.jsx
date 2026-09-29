import { useEffect, useState } from 'react'
import { api } from '../services/api'

export default function SearchBox({ onSelect }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])

  useEffect(() => {
    const text = query.trim()
    if (text.length < 2) {
      setResults([])
      return
    }
    let cancelled = false
    const timer = setTimeout(() => {
      api.search(text)
        .then((body) => !cancelled && setResults(body.results))
        .catch(() => !cancelled && setResults([]))
    }, 250)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [query])

  const choose = (id) => {
    onSelect(id)
    setResults([])
  }

  return (
    <div className="search">
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => event.key === 'Enter' && results[0] && choose(results[0].neuron_id)}
        placeholder="Neuron ID or cell type (e.g. DNa02)"
        aria-label="Search neurons"
      />
      {results.length > 0 && (
        <ul className="results">
          {results.map((result) => (
            <li key={result.neuron_id} onClick={() => choose(result.neuron_id)}>
              <span>{result.cell_type}</span>
              <small>{result.super_class} · {result.side}</small>
              <code>{result.neuron_id}</code>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
