import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useRef } from "react";
import * as THREE from "three";

function Hemi({ score }: { score: number }) {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, dt) => { if (ref.current) ref.current.rotation.y += dt * 0.3; });
  const color = score >= 60 ? "#1A7A4A" : score >= 40 ? "#C47B00" : score >= 25 ? "#D9534F" : "#C0392B";
  return (
    <group ref={ref}>
      <mesh>
        <sphereGeometry args={[1.2, 48, 48, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color={color} metalness={0.2} roughness={0.45} transparent opacity={0.9} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, score / 100 * 1.2, 0]}>
        <torusGeometry args={[1.21, 0.01, 8, 64]} />
        <meshBasicMaterial color="#1E3A5F" />
      </mesh>
    </group>
  );
}

export function ScoreHemisphere({ score }: { score: number }) {
  return (
    <div className="w-full h-[260px]">
      <Canvas camera={{ position: [0, 1.5, 3], fov: 45 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          <ambientLight intensity={0.7} />
          <pointLight position={[3, 4, 3]} intensity={1.1} />
          <pointLight position={[-3, 2, -2]} intensity={0.5} color="#1A7A4A" />
          <Hemi score={score} />
        </Suspense>
      </Canvas>
    </div>
  );
}
