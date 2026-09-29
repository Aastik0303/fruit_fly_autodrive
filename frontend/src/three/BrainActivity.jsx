import { useEffect, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import FitCamera from './FitCamera'
import NeuronCloud from './NeuronCloud'

const MAX_ACTIVE = 4000
const POSITIVE = new THREE.Color('#ffb03b')
const NEGATIVE = new THREE.Color('#4aa8ff')
export const PUSH_LEFT = '#7df9ff'
export const PUSH_RIGHT = '#ff7aa2'

function positionOf(cloud, indexByNode, nodeIndex) {
  const i = indexByNode.get(nodeIndex)
  return i === undefined ? null : [cloud.positions[i * 3], cloud.positions[i * 3 + 1], cloud.positions[i * 3 + 2]]
}

function ActivityPoints({ cloud, indexByNode, brain }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_ACTIVE * 3), 3))
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX_ACTIVE * 3), 3))
    g.setDrawRange(0, 0)
    return g
  }, [])
  useEffect(() => () => geometry.dispose(), [geometry])

  useEffect(() => {
    if (!brain) return
    const positions = geometry.attributes.position.array
    const colors = geometry.attributes.color.array
    const colour = new THREE.Color()
    let count = 0
    for (let k = 0; k < brain.nodes.length && count < MAX_ACTIVE; k++) {
      const p = positionOf(cloud, indexByNode, brain.nodes[k])
      if (!p) continue
      const value = brain.values[k] / 100
      colour.copy(value >= 0 ? POSITIVE : NEGATIVE).multiplyScalar(0.3 + 0.7 * Math.min(1, Math.abs(value)))
      positions.set(p, count * 3)
      colors.set([colour.r, colour.g, colour.b], count * 3)
      count++
    }
    geometry.setDrawRange(0, count)
    geometry.attributes.position.needsUpdate = true
    geometry.attributes.color.needsUpdate = true
  }, [brain, cloud, indexByNode, geometry])

  return (
    <points geometry={geometry} frustumCulled={false}>
      <pointsMaterial size={4} vertexColors sizeAttenuation transparent depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  )
}

function DriverMarkers({ cloud, indexByNode, drivers }) {
  return (drivers ?? []).slice(0, 5).map((driver) => {
    const p = positionOf(cloud, indexByNode, driver.node_index)
    if (!p) return null
    return (
      <mesh key={driver.neuron_id} position={p}>
        <sphereGeometry args={[7, 16, 16]} />
        <meshBasicMaterial color={driver.push > 0 ? PUSH_LEFT : PUSH_RIGHT} transparent opacity={0.9} />
      </mesh>
    )
  })
}

export default function BrainActivity({ cloud, decision }) {
  const indexByNode = useMemo(() => new Map(cloud.node_index.map((node, i) => [node, i])), [cloud])
  return (
    <Canvas camera={{ position: [0, 0, 1000], fov: 50, near: 1, far: 5000 }} dpr={[1, 2]}>
      <color attach="background" args={['#0b0e14']} />
      <FitCamera width={880} height={520} />
      <NeuronCloud cloud={cloud} dimmed onPick={() => {}} />
      <ActivityPoints cloud={cloud} indexByNode={indexByNode} brain={decision?.brain} />
      <DriverMarkers cloud={cloud} indexByNode={indexByNode} drivers={decision?.drivers} />
      <OrbitControls makeDefault enableDamping />
    </Canvas>
  )
}
