export function connectGame({ onMessage, onStatus }) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const socket = new WebSocket(`${protocol}://${window.location.host}/api/ws/game`)

  socket.onopen = () => onStatus('connected')
  socket.onclose = () => onStatus('disconnected')
  socket.onerror = () => onStatus('error')
  socket.onmessage = (event) => onMessage(JSON.parse(event.data))

  return {
    send: (command) => {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command))
    },
    close: () => socket.close(),
  }
}
