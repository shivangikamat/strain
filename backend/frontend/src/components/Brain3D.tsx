/**
 * Anatomical mesh: brain_sliced.glb from https://github.com/pachoclo/lebrain-threejs
 * (MIT-licensed demo project; verify upstream model terms for production redistribution).
 * Loaded from jsDelivr for zero repo binary weight; replace with /public/models/... for offline.
 */
import { Suspense, useLayoutEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Environment, OrbitControls, useGLTF } from '@react-three/drei'
import * as THREE from 'three'

/** Vite: place file at `public/models/brain_sliced.glb` or use a full URL. */
const BRAIN_GLB_URL = '/models/brain_sliced.glb'

useGLTF.preload(BRAIN_GLB_URL)

// 14 Channels — DREAMER / EMOTIV EPOC schematic directions (unit-ish); `scalpPosition` projects onto a small sphere inside the mesh.
const ELECTRODE_DIRECTIONS: Record<string, [number, number, number]> = {
  AF3: [-0.35, 0.85, 0.45],
  AF4: [0.35, 0.85, 0.45],
  F7: [-0.82, 0.48, 0.38],
  F8: [0.82, 0.48, 0.38],
  F3: [-0.45, 0.62, 0.68],
  F4: [0.45, 0.62, 0.68],
  FC5: [-0.88, 0.28, 0.48],
  FC6: [0.88, 0.28, 0.48],
  T7: [-0.95, 0.02, 0.05],
  T8: [0.95, 0.02, 0.05],
  P7: [-0.72, -0.58, 0.38],
  P8: [0.72, -0.58, 0.38],
  O1: [-0.32, -0.82, 0.32],
  O2: [0.32, -0.82, 0.32],
}

/** Map schematic directions onto a sphere fully inside the scaled cortical hull (~max dim 2.35). */
function scalpPosition(dir: [number, number, number]): [number, number, number] {
  const [x, y, z] = dir
  const len = Math.sqrt(x * x + y * y + z * z) || 1
  const r = 0.52
  return [(x / len) * r, (y / len) * r, (z / len) * r]
}

function ElectrodeMarker({ position, value }: { position: [number, number, number]; value: number }) {
  const meshRef = useRef<THREE.Mesh>(null)

  const color = useMemo(() => {
    if (value > 2.0) return '#ef4444'
    if (value > 1.2) return '#f97316'
    if (value > 0.6) return '#eab308'
    return '#22c55e'
  }, [value])

  useFrame(({ clock }) => {
    if (meshRef.current) {
      const scale = 1.0 + Math.sin(clock.elapsedTime * value * 3) * 0.1
      meshRef.current.scale.set(scale, scale, scale)
    }
  })

  return (
    <mesh position={position} ref={meshRef} renderOrder={2}>
      <sphereGeometry args={[0.048, 18, 18]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={Math.min(value * 0.45, 1.2)}
        toneMapped={false}
        depthTest
        depthWrite
      />
    </mesh>
  )
}

/** Translucent cortical shell so emissive electrodes read through the volume. */
function AnatomicalBrain() {
  const { scene } = useGLTF(BRAIN_GLB_URL)
  const root = useRef<THREE.Group>(null)
  const shell = useMemo(() => {
    const g = scene.clone(true)
    g.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      const prev = child.material
      const prevArr = Array.isArray(prev) ? prev : [prev]
      const nextMats = prevArr.map(
        () =>
          new THREE.MeshPhysicalMaterial({
            color: new THREE.Color('#c7d7f0'),
            emissive: new THREE.Color('#312e81'),
            emissiveIntensity: 0.06,
            transparent: true,
            opacity: 0.14,
            transmission: 0.78,
            thickness: 1.15,
            roughness: 0.22,
            metalness: 0.05,
            ior: 1.38,
            side: THREE.DoubleSide,
            depthWrite: false,
            attenuationColor: new THREE.Color('#4f46e5'),
            attenuationDistance: 1.25,
            clearcoat: 0.35,
            clearcoatRoughness: 0.4,
          }),
      )
      child.material = nextMats.length === 1 ? nextMats[0]! : nextMats
      child.castShadow = false
      child.receiveShadow = false
      child.renderOrder = 0
      for (const m of prevArr) {
        if (m && typeof (m as THREE.Material).dispose === 'function') (m as THREE.Material).dispose()
      }
    })
    return g
  }, [scene])

  const shellMirrored = useMemo(() => {
    const m = shell.clone(true)
    m.scale.set(-1, 1, 1)
    m.updateMatrixWorld(true)
    return m
  }, [shell])

  useLayoutEffect(() => {
    const g = root.current
    if (!g) return
    g.scale.setScalar(1)
    g.position.set(0, 0, 0)
    g.rotation.set(0, 0, 0)
    g.updateMatrixWorld(true)
    const box = new THREE.Box3().setFromObject(g)
    const size = new THREE.Vector3()
    box.getSize(size)
    const max = Math.max(size.x, size.y, size.z, 1e-6)
    const target = 2.35
    g.scale.setScalar(target / max)
    g.updateMatrixWorld(true)
    const box2 = new THREE.Box3().setFromObject(g)
    const c = new THREE.Vector3()
    box2.getCenter(c)
    g.position.sub(c)
  }, [shell])

  return (
    <group ref={root}>
      <primitive object={shell} />
      {/* Mirror half-brain across YZ plane to approximate a full cerebrum */}
      <primitive object={shellMirrored} />
    </group>
  )
}

function BrainFallback() {
  return (
    <mesh renderOrder={0}>
      <icosahedronGeometry args={[1.05, 2]} />
      <meshPhysicalMaterial
        color="#e2e8f0"
        transparent
        opacity={0.12}
        transmission={0.85}
        thickness={0.6}
        roughness={0.35}
        wireframe
        depthWrite={false}
      />
    </mesh>
  )
}

function SceneContent({ bandMeanPower }: { bandMeanPower: Record<string, number> | undefined }) {
  const { gl } = useThree()
  useLayoutEffect(() => {
    gl.setClearColor(0x000000, 0)
  }, [gl])

  return (
    <>
      <ambientLight intensity={0.38} />
      <directionalLight position={[4, 6, 3]} intensity={0.85} color="#f8fafc" />
      <directionalLight position={[-4, -2, -2]} intensity={0.35} color="#6366f1" />
      <pointLight position={[0, 1.2, 1.8]} intensity={0.5} color="#93c5fd" />

      <Suspense fallback={<BrainFallback />}>
        <Environment preset="night" environmentIntensity={0.55} />
        <AnatomicalBrain />
      </Suspense>

      {Object.entries(ELECTRODE_DIRECTIONS).map(([name, dir]) => {
        const powerLevel = bandMeanPower
          ? bandMeanPower[`beta_${name}`] || bandMeanPower[`theta_${name}`] || 0.5
          : 0.5
        return <ElectrodeMarker key={name} position={scalpPosition(dir)} value={powerLevel} />
      })}

      <OrbitControls enableZoom enablePan={false} autoRotate autoRotateSpeed={1.2} maxDistance={6} minDistance={1.8} />
    </>
  )
}

export function Brain3D({ bandMeanPower }: { bandMeanPower: Record<string, number> | undefined }) {
  return (
    <div
      style={{
        width: '100%',
        maxWidth: '100%',
        minWidth: 0,
        height: '350px',
        boxSizing: 'border-box',
        background: 'rgba(0,0,0,0.2)',
        borderRadius: '16px',
        border: '1px solid rgba(255,255,255,0.08)',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <Canvas
        camera={{ position: [0, 0.15, 3.25], fov: 42 }}
        gl={{ alpha: true, antialias: true, powerPreference: 'high-performance' }}
        style={{ display: 'block', width: '100%', height: '100%', maxWidth: '100%' }}
        dpr={[1, 2]}
      >
        <SceneContent bandMeanPower={bandMeanPower} />
      </Canvas>
    </div>
  )
}
