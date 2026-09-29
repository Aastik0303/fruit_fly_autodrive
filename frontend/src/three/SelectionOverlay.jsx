import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { ROLE_COLORS } from '../colors'

function positionOf(cloud, indexByNode, nodeIndex) {
  const i = indexByNode.get(nodeIndex)
  if (i === undefined) return null
  return [cloud.positions[i * 3], cloud.positions[i * 3 + 1], cloud.positions[i * 3 + 2]]
}

// Selected neuron as a white sphere, its partners as larger dots, and one line per connection.
// Line brightness grows with synapse count.
export default function SelectionOverlay({ cloud, indexByNode, neuron, connections }) {
  const { center, lines, partners } = useMemo(() => {
    const center = positionOf(cloud, indexByNode, neuron.node_index)
    const linePositions = []
    const lineColors = []
    const partnerPositions = []
    const partnerColors = []
    const groups = [
      [connections.incoming?.items ?? [], new THREE.Color(ROLE_COLORS.incoming)],
      [connections.outgoing?.items ?? [], new THREE.Color(ROLE_COLORS.outgoing)],
    ]
    const strongest = Math.max(1, ...groups.flatMap(([items]) => items.map((item) => item.synapse_count)))

    for (const [items, base] of groups) {
      for (const item of items) {
        const p = positionOf(cloud, indexByNode, item.node_index)
        if (!p) continue
        partnerPositions.push(...p)
        partnerColors.push(base.r, base.g, base.b)
        if (center) {
          const c = base.clone().multiplyScalar(0.25 + 0.75 * Math.sqrt(item.synapse_count / strongest))
          linePositions.push(...center, ...p)
          lineColors.push(c.r, c.g, c.b, c.r, c.g, c.b)
        }
      }
    }

    const lines = new THREE.BufferGeometry()
    lines.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3))
    lines.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3))
    const partners = new THREE.BufferGeometry()
    partners.setAttribute('position', new THREE.Float32BufferAttribute(partnerPositions, 3))
    partners.setAttribute('color', new THREE.Float32BufferAttribute(partnerColors, 3))
    return { center, lines, partners }
  }, [cloud, indexByNode, neuron, connections])

  useEffect(() => () => {
    lines.dispose()
    partners.dispose()
  }, [lines, partners])

  return (
    <group>
      <lineSegments geometry={lines}>
        <lineBasicMaterial vertexColors transparent opacity={0.85} />
      </lineSegments>
      <points geometry={partners}>
        <pointsMaterial size={7} vertexColors sizeAttenuation />
      </points>
      {center && (
        <mesh position={center}>
          <sphereGeometry args={[6, 20, 20]} />
          <meshBasicMaterial color={ROLE_COLORS.selected} />
        </mesh>
      )}
    </group>
  )
}
