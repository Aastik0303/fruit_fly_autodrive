import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// A stylised Drosophila seen from above: head pointing up the screen (+y), back facing the camera (+z).
const BODY = '#4f3d2b'
const ABDOMEN = '#a07c45'
const STRIPE = '#3b2b1b'
const EYE = '#b3261e'
const LEG = '#2e241a'
const WING = '#d8ecff'

const LEG_ANGLES = [0.75, 0, -0.75] // front, middle, hind (radians from sideways)

function Legs() {
  return [-1, 1].flatMap((side) =>
    LEG_ANGLES.map((angle, i) => {
      const length = i === 2 ? 1.9 : 1.6
      const direction = [side * Math.cos(angle), Math.sin(angle)]
      const root = [side * 0.4, 0.3 - i * 0.3]
      return (
        <mesh
          key={`${side}-${i}`}
          position={[root[0] + (direction[0] * length) / 2, root[1] + (direction[1] * length) / 2, -0.35]}
          rotation={[0, 0, side > 0 ? angle - Math.PI / 2 : Math.PI / 2 - angle]}
        >
          <cylinderGeometry args={[0.035, 0.06, length, 6]} />
          <meshStandardMaterial color={LEG} />
        </mesh>
      )
    }),
  )
}

function Wing({ side }) {
  const pivot = useRef()
  useFrame(({ clock }) => {
    const beat = 0.5 + 0.5 * Math.sin(clock.elapsedTime * 70 + (side > 0 ? 0 : Math.PI * 0.05))
    pivot.current.rotation.z = side * (0.2 + 0.7 * beat)
    pivot.current.rotation.y = side * 0.35 * (beat - 0.5)
  })
  return (
    <group ref={pivot} position={[side * 0.25, 0.3, 0.55]}>
      <mesh position={[side * 0.35, -1.15, 0]} rotation={[0, 0, side * 0.25]} scale={[0.55, 1.45, 1]}>
        <circleGeometry args={[1, 32]} />
        <meshPhysicalMaterial color={WING} transparent opacity={0.35} roughness={0.15} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
    </group>
  )
}

export default function Fly() {
  return (
    <group scale={0.85}>
      <Legs />

      {/* abdomen with dark bands */}
      <mesh position={[0, -1.45, -0.05]} scale={[0.95, 1.55, 0.8]}>
        <sphereGeometry args={[0.62, 24, 24]} />
        <meshStandardMaterial color={ABDOMEN} roughness={0.6} />
      </mesh>
      {[[-1.0, 0.56], [-1.5, 0.58], [-2.0, 0.46]].map(([y, radius]) => (
        <mesh key={y} position={[0, y, -0.05]} rotation={[Math.PI / 2, 0, 0]} scale={[1, 0.8, 1]}>
          <torusGeometry args={[radius, 0.07, 8, 32]} />
          <meshStandardMaterial color={STRIPE} />
        </mesh>
      ))}

      {/* thorax and head */}
      <mesh position={[0, 0.1, 0]} scale={[1, 1.2, 0.85]}>
        <sphereGeometry args={[0.62, 24, 24]} />
        <meshStandardMaterial color={BODY} roughness={0.7} />
      </mesh>
      <mesh position={[0, 1.15, 0]}>
        <sphereGeometry args={[0.5, 24, 24]} />
        <meshStandardMaterial color={BODY} roughness={0.7} />
      </mesh>

      {/* compound eyes and antennae */}
      {[-1, 1].map((side) => (
        <group key={side}>
          <mesh position={[side * 0.38, 1.25, 0.15]} scale={[0.9, 1.1, 1]}>
            <sphereGeometry args={[0.36, 24, 24]} />
            <meshStandardMaterial color={EYE} roughness={0.35} emissive="#3a0000" />
          </mesh>
          <mesh position={[side * 0.14, 1.62, 0.35]}>
            <sphereGeometry args={[0.08, 8, 8]} />
            <meshStandardMaterial color={LEG} />
          </mesh>
        </group>
      ))}

      <Wing side={-1} />
      <Wing side={1} />
    </group>
  )
}
