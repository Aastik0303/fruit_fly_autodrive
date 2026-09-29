import { useState } from 'react'
import ExplorerPage from './pages/ExplorerPage'
import FlyGamePage from './pages/FlyGamePage'

const TABS = [
  ['explorer', 'Connectome Explorer'],
  ['game', 'Fly plays Dodge'],
]

export default function App() {
  const [tab, setTab] = useState(() => (window.location.hash === '#game' ? 'game' : 'explorer'))

  const select = (id) => {
    setTab(id)
    window.location.hash = id
  }

  return (
    <div className="app">
      <nav className="topbar">
        <span className="brand">NeuroPath AI</span>
        {TABS.map(([id, label]) => (
          <button key={id} type="button" className={`tab ${tab === id ? 'active' : ''}`} onClick={() => select(id)}>
            {label}
          </button>
        ))}
      </nav>
      {tab === 'explorer' ? <ExplorerPage /> : <FlyGamePage />}
    </div>
  )
}
