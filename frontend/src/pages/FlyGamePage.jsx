import { useEffect, useRef, useState } from 'react'
import DriversList from '../components/game/DriversList'
import EyeView from '../components/game/EyeView'
import PipelineStrip from '../components/game/PipelineStrip'
import QBars from '../components/game/QBars'
import ReportTable from '../components/game/ReportTable'
import { api } from '../services/api'
import { connectGame } from '../services/gameSocket'
import BrainActivity from '../three/BrainActivity'
import GameArena from '../three/GameArena'

const EMPTY_HUD = { score: 0, attempts: 3, milestone: null, done: false }

export default function FlyGamePage() {
  const frameRef = useRef(null) // latest frame, read by the 3D render loop without re-rendering React
  const socketRef = useRef(null)
  const [hello, setHello] = useState(null)
  const [hud, setHud] = useState(EMPTY_HUD)
  const [decision, setDecision] = useState(null)
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('connecting')
  const [error, setError] = useState(null)
  const [speed, setSpeed] = useState(1)
  const [policy, setPolicy] = useState('connectome')
  const [cloud, setCloud] = useState(null)

  useEffect(() => {
    api.pointCloud().then(setCloud).catch((err) => setError(`Brain positions not available: ${err.message}`))
  }, [])

  useEffect(() => {
    const socket = connectGame({
      onStatus: setStatus,
      onMessage: (message) => {
        if (message.type === 'hello') setHello(message)
        if (message.type === 'error') setError(message.message)
        if (message.type !== 'frame') return
        frameRef.current = message
        if (message.decision) setDecision(message.decision)
        setRunning(message.running)
        setHud((previous) =>
          previous.score === message.score && previous.attempts === message.attempts &&
          previous.milestone === message.milestone && previous.done === message.done
            ? previous
            : { score: message.score, attempts: message.attempts, milestone: message.milestone, done: message.done },
        )
      },
    })
    socketRef.current = socket
    return () => socket.close()
  }, [])

  const send = (command) => socketRef.current?.send(command)
  const restart = () => {
    send({ cmd: 'reset' })
    send({ cmd: 'start' })
  }
  const configure = (changes) => send({ cmd: 'config', speed, policy, ...changes })

  const ready = status === 'connected' && hello

  return (
    <div className="game-page">
      <section className="game-stage">
        <GameArena frameRef={frameRef} />
        <div className="hud">
          <span>Score <strong>{hud.score}</strong></span>
          <span>Attempts <strong>{hud.attempts}</strong></span>
          <span>Move <strong>{decision?.action ?? '—'}</strong></span>
        </div>
        {hud.milestone && <div className="milestone">{hud.milestone}</div>}
        {hud.done && (
          <div className="game-over">
            <h2>Game Over</h2>
            <p>Final score: {hud.score}</p>
            <button type="button" className="primary" onClick={restart}>Replay</button>
          </div>
        )}
        <div className="controls">
          {running ? (
            <button type="button" onClick={() => send({ cmd: 'pause' })}>Pause</button>
          ) : (
            <button type="button" className="primary" disabled={!ready} onClick={() => send({ cmd: 'start' })}>
              {ready ? 'Start' : 'Connecting…'}
            </button>
          )}
          <button type="button" disabled={!ready} onClick={restart}>Restart</button>
          <label>
            Speed
            <select value={speed} onChange={(event) => { const value = Number(event.target.value); setSpeed(value); configure({ speed: value }) }}>
              {(hello?.speeds ?? [1, 2, 4, 8]).map((value) => <option key={value} value={value}>{value}×</option>)}
            </select>
          </label>
          <label>
            Controller
            <select value={policy} onChange={(event) => { setPolicy(event.target.value); configure({ policy: event.target.value }) }}>
              {(hello?.policies ?? []).map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
        </div>
        {error && <div className="error floating">{error}</div>}
      </section>

      <section className="brain-stage">
        <div className="stage-title">
          Brain activity <span>orange = net excitatory drive · blue = net inhibitory · big dots = top descending neurons</span>
        </div>
        {cloud ? <BrainActivity cloud={cloud} decision={decision} /> : <div className="loading">Loading brain…</div>}
      </section>

      <aside className="game-panels">
        <PipelineStrip hello={hello} />
        <QBars decision={decision} />
        <EyeView view={decision?.view} />
        <DriversList drivers={decision?.drivers} />
        <ReportTable report={hello?.report} />
      </aside>
    </div>
  )
}
