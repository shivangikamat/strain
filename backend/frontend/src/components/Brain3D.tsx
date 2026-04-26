import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// 14 Channels specified in DREAMER (EMOTIV EPOC layout)
const ELECTRODE_POSITIONS: Record<string, [number, number, number]> = {
  AF3: [-0.3, 0.8, 0.5],
  AF4: [0.3, 0.8, 0.5],
  F7: [-0.7, 0.5, 0.4],
  F8: [0.7, 0.5, 0.4],
  F3: [-0.4, 0.6, 0.7],
  F4: [0.4, 0.6, 0.7],
  FC5: [-0.8, 0.3, 0.5],
  FC6: [0.8, 0.3, 0.5],
  T7: [-0.9, 0.0, 0.0],
  T8: [0.9, 0.0, 0.0],
  P7: [-0.7, -0.6, 0.4],
  P8: [0.7, -0.6, 0.4],
  O1: [-0.3, -0.8, 0.3],
  O2: [0.3, -0.8, 0.3],
};

function ElectrodeMarker({ position, value, name }: { position: [number, number, number], value: number, name: string }) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Map arbitrary power (0.0 to maybe 5.0) to local intense thresholds
  // Fall back heavily to green-yellow if not super severe, red if severe.
  const color = useMemo(() => {
    // Standardize 'value' since PSD values are highly variable.
    // Assuming values range roughly 0-4
    if (value > 2.0) return '#ef4444'; // Red (High Anxiety/Beta)
    if (value > 1.2) return '#f97316'; // Orange
    if (value > 0.6) return '#eab308'; // Yellow
    return '#22c55e'; // Green (Calm)
  }, [value]);

  // Make them pulse slightly based on value
  useFrame(({ clock }) => {
    if (meshRef.current) {
       const scale = 1.0 + Math.sin(clock.elapsedTime * value * 3) * 0.1;
       meshRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <mesh position={position} ref={meshRef}>
      <sphereGeometry args={[0.07, 16, 16]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={Math.min(value * 0.4, 1)} />
    </mesh>
  );
}

export function Brain3D({ bandMeanPower }: { bandMeanPower: Record<string, number> | undefined }) {
  return (
    <div style={{ width: '100%', height: '350px', background: 'rgba(0,0,0,0.2)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.08)', overflow: 'hidden', position: 'relative' }}>
      <Canvas camera={{ position: [0, 0, 3.5] }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4f46e5" />
        
        {/* Simplified translucent brain sphere for the background */}
        <mesh>
          <sphereGeometry args={[1.2, 32, 32]} />
          <meshPhysicalMaterial 
             color="#f8fafc" 
             transparent 
             opacity={0.15} 
             roughness={0.1}
             transmission={0.9} 
             thickness={0.5} 
          />
        </mesh>

        {Object.entries(ELECTRODE_POSITIONS).map(([name, pos]) => {
          // Look for 'beta_<channel>' power as the primary indicator for excitement/anxiety mapping
          const powerLevel = bandMeanPower ? (bandMeanPower[`beta_${name}`] || bandMeanPower[`theta_${name}`] || 0.5) : 0.5;
          
          return (
            <ElectrodeMarker
              key={name}
              position={pos}
              value={powerLevel}
              name={name}
            />
          );
        })}
        
        <OrbitControls enableZoom={true} enablePan={false} autoRotate autoRotateSpeed={1.5} />
      </Canvas>
    </div>
  );
}
