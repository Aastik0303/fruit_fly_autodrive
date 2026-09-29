import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'

// Places the perspective camera on the z axis so that a width x height rectangle centred at the origin
// fits the canvas, whatever the panel's aspect ratio. Re-runs when the canvas is resized.
export default function FitCamera({ width, height, margin = 1.05 }) {
  const camera = useThree((state) => state.camera)
  const size = useThree((state) => state.size)
  const controls = useThree((state) => state.controls)

  useEffect(() => {
    const aspect = size.width / size.height
    const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))
    const distance = (margin * Math.max(height / 2, width / 2 / aspect)) / tanHalfFov
    camera.position.set(0, 0, distance)
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()
    if (controls) {
      controls.target.set(0, 0, 0)
      controls.update()
    }
  }, [camera, controls, size.width, size.height, width, height, margin])

  return null
}
