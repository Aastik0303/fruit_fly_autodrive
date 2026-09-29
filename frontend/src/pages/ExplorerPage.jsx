import { useEffect, useMemo, useState } from 'react'
import Legend from '../components/Legend'
import NeuronPanel from '../components/NeuronPanel'
import SearchBox from '../components/SearchBox'
import { api } from '../services/api'
import BrainScene from '../three/BrainScene'

export default function ExplorerPage() {
  const [cloud, setCloud] = useState(null)
  const [stats, setStats] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [details, setDetails] = useState(null)
  const [minSynapses, setMinSynapses] = useState(5)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([api.pointCloud(), api.stats()])
      .then(([pointCloud, summary]) => {
        setCloud(pointCloud)
        setStats(summary)
      })
      .catch((err) => setError(`Backend not reachable: ${err.message}`))
  }, [])

  // node_index -> position in the point arrays
  const indexByNode = useMemo(
    () => new Map((cloud?.node_index ?? []).map((node, i) => [node, i])),
    [cloud],
  )

  useEffect(() => {
    if (!selectedId) {
      setDetails(null)
      return
    }
    let cancelled = false
    setError(null)
    Promise.all([api.neuron(selectedId), api.connections(selectedId, minSynapses, 50)])
      .then(([neuron, connections]) => !cancelled && setDetails({ neuron, connections }))
      .catch((err) => !cancelled && setError(err.message))
    return () => {
      cancelled = true
    }
  }, [selectedId, minSynapses])

  return (
    <div className="layout">
      <aside className="sidebar">
        <header>
          <h1>NeuroPath AI</h1>
          <p>
            FlyWire FAFB v783
            {stats && ` · ${stats.neurons.toLocaleString()} neurons · ${stats.edges.toLocaleString()} connections`}
          </p>
        </header>

        <SearchBox onSelect={setSelectedId} />
        {error && <div className="error">{error}</div>}

        {details ? (
          <NeuronPanel
            neuron={details.neuron}
            connections={details.connections}
            minSynapses={minSynapses}
            onMinSynapses={setMinSynapses}
            onSelect={setSelectedId}
            onClear={() => setSelectedId(null)}
          />
        ) : (
          <p className="hint">Search for a neuron ID or cell type, or click any point in the brain.</p>
        )}

        <Legend classes={cloud?.classes} counts={stats?.super_class_counts} />
      </aside>

      <main className="viewport">
        {cloud ? (
          <>
            <BrainScene
              cloud={cloud}
              indexByNode={indexByNode}
              details={details}
              onPick={(pointIndex) => setSelectedId(cloud.neuron_ids[pointIndex])}
            />
            <div className="viewport-hint">drag to rotate · right-drag to pan · scroll to zoom · click a neuron</div>
          </>
        ) : (
          <div className="loading">{error ? 'Start the backend and reload.' : 'Loading neuron positions…'}</div>
        )}
      </main>
    </div>
  )
}
