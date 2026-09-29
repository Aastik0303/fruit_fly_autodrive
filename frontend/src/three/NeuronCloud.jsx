import { useEffect, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { CLASS_COLORS } from '../colors'

// All ~139k neurons drawn as one THREE.Points object: a single GPU draw call.
export default function NeuronCloud({ cloud, dimmed, onPick }) {
  const raycaster = useThree((state) => state.raycaster)
  useEffect(() => {
    raycaster.params.Points.threshold = 2.5 // click tolerance in micrometres
  }, [raycaster])

  const geometry = useMemo(() => {
    const palette = cloud.classes.map((name) => new THREE.Color(CLASS_COLORS[name] ?? CLASS_COLORS.unknown))
    const colors = new Float32Array(cloud.count * 3)
    cloud.class_code.forEach((code, i) => {
      colors.set([palette[code].r, palette[code].g, palette[code].b], i * 3)
    })
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(cloud.positions, 3))
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    g.computeBoundingSphere()
    return g
  }, [cloud])

  useEffect(() => () => geometry.dispose(), [geometry])

  const handleClick = (event) => {
    if (event.delta > 4) return // the pointer moved: it was a rotate drag, not a click
    event.stopPropagation()
    onPick(event.index)
  }

  return (
    <points geometry={geometry} onClick={handleClick}>
      <pointsMaterial size={2.5} vertexColors sizeAttenuation transparent opacity={dimmed ? 0.12 : 0.75} depthWrite={false} />
    </points>
  )
}
