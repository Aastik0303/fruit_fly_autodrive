import { useEffect, useRef } from 'react'

export default function EyeView({ view }) {
  const canvasRef = useRef()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !view) return
    const context = canvas.getContext('2d')
    const rows = view.length
    const cols = view[0].length
    const cellWidth = canvas.width / cols
    const cellHeight = canvas.height / rows
    context.clearRect(0, 0, canvas.width, canvas.height)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const rgb = c < cols / 2 ? '125, 205, 255' : '255, 170, 110'
        context.fillStyle = `rgba(${rgb}, ${0.08 + 0.92 * view[r][c]})`
        context.fillRect(c * cellWidth + 1, r * cellHeight + 1, cellWidth - 2, cellHeight - 2)
      }
    }
    context.fillStyle = '#ffffff'
    context.fillRect(canvas.width / 2 - 5, canvas.height - 7, 10, 7)
  }, [view])

  return (
    <section className="card">
      <h3>What the eyes receive</h3>
      <canvas ref={canvasRef} width={320} height={192} className="eye-canvas" />
      <p className="hint">
        20 × 12 grid centred on the fly (white mark). Blue half → left-eye R7/R8 photoreceptors,
        orange half → right eye. Brighter = more of that cell covered by a block.
      </p>
    </section>
  )
}
