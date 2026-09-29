import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import NeuronCloud from './NeuronCloud'
import SelectionOverlay from './SelectionOverlay'

export default function BrainScene({ cloud, indexByNode, details, onPick }) {
  return (
    <Canvas camera={{ position: [0, 0, 950], fov: 50, near: 1, far: 5000 }} dpr={[1, 2]}>
      <color attach="background" args={['#0b0e14']} />
      <NeuronCloud cloud={cloud} dimmed={Boolean(details)} onPick={onPick} />
      {details && (
        <SelectionOverlay
          cloud={cloud}
          indexByNode={indexByNode}
          neuron={details.neuron}
          connections={details.connections}
        />
      )}
      <OrbitControls makeDefault enableDamping />
    </Canvas>
  )
}
