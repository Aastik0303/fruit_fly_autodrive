import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import FitCamera from './FitCamera'
import Fly from './Fly'

const WIDTH = 1000
const HEIGHT = 600
const PLAYER = 30
const BLOCK = 50
const MAX_BLOCKS = 10
const SCALE = 0.1 // world units per game pixel

const toWorldX = (x) => (x - WIDTH / 2) * SCALE
const toWorldY = (y) => (HEIGHT / 2 - y) * SCALE

// background follows the original game: yellow -> orange -> red as attempts run out (toned down for a dark UI)
const BACKDROP = { 3: new THREE.Color('#3f3b1b'), 2: new THREE.Color('#4a3113'), 1: new THREE.Color('#4d1b1b'), 0: new THREE.Color('#4d1b1b') }
const HIT_FLASH = new THREE.Color('#ff3b3b')

function GridLines() {
  const geometry = useMemo(() => {
    const points = []
    for (let x = 0; x <= WIDTH; x += 50) points.push(toWorldX(x), toWorldY(0), 0, toWorldX(x), toWorldY(HEIGHT), 0)
    for (let y = 0; y <= HEIGHT; y += 50) points.push(toWorldX(0), toWorldY(y), 0, toWorldX(WIDTH), toWorldY(y), 0)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
    return g
  }, [])
  return (
    <lineSegments geometry={geometry} position={[0, 0, -2.9]}>
      <lineBasicMaterial color="#ffffff" transparent opacity={0.06} />
    </lineSegments>
  )
}

function Scene({ frameRef }) {
  const fly = useRef()
  const blocks = useRef([])
  const backdrop = useRef()
  const flash = useRef(0)
  const colour = useMemo(() => new THREE.Color(), [])

  useFrame((_, delta) => {
    const frame = frameRef.current
    if (!frame || !fly.current) return

    const targetX = toWorldX(frame.player_x + PLAYER / 2)
    const currentX = fly.current.position.x
    fly.current.position.x = THREE.MathUtils.damp(currentX, targetX, 25, delta)
    fly.current.position.y = toWorldY(frame.player_y + PLAYER / 2)
    const bank = THREE.MathUtils.clamp((currentX - targetX) * 0.4, -0.5, 0.5)
    fly.current.rotation.y = THREE.MathUtils.damp(fly.current.rotation.y, -bank, 10, delta)
    fly.current.rotation.z = THREE.MathUtils.damp(fly.current.rotation.z, bank * 0.4, 10, delta)

    blocks.current.forEach((mesh, i) => {
      const enemy = frame.enemies[i]
      mesh.visible = Boolean(enemy)
      if (enemy) mesh.position.set(toWorldX(enemy[0] + BLOCK / 2), toWorldY(enemy[1] + BLOCK / 2), 0)
    })

    if (frame.hit) flash.current = 1
    flash.current = Math.max(0, flash.current - delta * 2.5)
    colour.copy(BACKDROP[frame.attempts] ?? BACKDROP[0]).lerp(HIT_FLASH, flash.current * 0.7)
    backdrop.current.color.copy(colour)
  })

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[20, 30, 60]} intensity={1.6} />

      <mesh position={[0, 0, -3]}>
        <planeGeometry args={[WIDTH * SCALE, HEIGHT * SCALE]} />
        <meshStandardMaterial ref={backdrop} color={BACKDROP[3]} roughness={1} />
      </mesh>
      <GridLines />
      {[-1, 1].map((side) => (
        <mesh key={side} position={[side * (WIDTH * SCALE / 2 + 0.4), 0, 0]}>
          <boxGeometry args={[0.8, HEIGHT * SCALE, 6]} />
          <meshStandardMaterial color="#20242e" />
        </mesh>
      ))}

      {Array.from({ length: MAX_BLOCKS }, (_, i) => (
        <mesh key={i} ref={(mesh) => { blocks.current[i] = mesh }} visible={false}>
          <boxGeometry args={[BLOCK * SCALE, BLOCK * SCALE, BLOCK * SCALE]} />
          <meshStandardMaterial color="#17171c" emissive="#101014" roughness={0.45} metalness={0.25} />
        </mesh>
      ))}

      <group ref={fly} position={[0, toWorldY(HEIGHT - PLAYER - 10 + PLAYER / 2), 1]}>
        <Fly />
      </group>
    </>
  )
}

export default function GameArena({ frameRef }) {
  return (
    <Canvas camera={{ position: [0, 0, 82], fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={['#0b0e14']} />
      {/* arena plus walls, with room for the HUD above and the controls below */}
      <FitCamera width={WIDTH * SCALE + 2} height={HEIGHT * SCALE + 14} />
      <Scene frameRef={frameRef} />
    </Canvas>
  )
}
