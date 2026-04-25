import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useMemo, useRef } from "react";
import * as THREE from "three";

function Globe() {
  const group = useRef<THREE.Group>(null);
  const nodesRef = useRef<THREE.Group>(null);

  const nodes = useMemo(() => [
    { name: "London", lat: 51.5, lon: -0.12 },
    { name: "NYC", lat: 40.7, lon: -74 },
    { name: "Tokyo", lat: 35.7, lon: 139.7 },
    { name: "Frankfurt", lat: 50.1, lon: 8.7 },
    { name: "Mumbai", lat: 19.1, lon: 72.9 },
    { name: "Singapore", lat: 1.35, lon: 103.8 },
  ], []);

  const R = 2;
  const toVec = (lat: number, lon: number) => {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    return new THREE.Vector3(
      -R * Math.sin(phi) * Math.cos(theta),
      R * Math.cos(phi),
      R * Math.sin(phi) * Math.sin(theta)
    );
  };

  useFrame((_, dt) => {
    if (group.current) group.current.rotation.y += dt * 0.08;
  });

  const arcs = useMemo(() => {
    const pairs: [THREE.Vector3, THREE.Vector3][] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (Math.random() > 0.4) pairs.push([toVec(nodes[i].lat, nodes[i].lon), toVec(nodes[j].lat, nodes[j].lon)]);
      }
    }
    return pairs.map(([a, b]) => {
      const mid = a.clone().add(b).multiplyScalar(0.5).normalize().multiplyScalar(R * 1.3);
      const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
      return new THREE.BufferGeometry().setFromPoints(curve.getPoints(40));
    });
  }, [nodes]);

  return (
    <group ref={group} rotation={[0.41, 0, 0]}>
      <mesh>
        <icosahedronGeometry args={[R, 4]} />
        <meshBasicMaterial color="#1A7A4A" wireframe transparent opacity={0.18} />
      </mesh>
      <mesh>
        <sphereGeometry args={[R * 0.99, 32, 32]} />
        <meshBasicMaterial color="#F7F9F7" transparent opacity={0.85} />
      </mesh>
      <group ref={nodesRef}>
        {nodes.map((n, i) => {
          const v = toVec(n.lat, n.lon);
          return (
            <mesh key={i} position={v}>
              <sphereGeometry args={[0.04, 12, 12]} />
              <meshBasicMaterial color="#1A7A4A" />
            </mesh>
          );
        })}
        {arcs.map((g, i) => (
          <line key={i}>
            <primitive object={g} attach="geometry" />
            <lineBasicMaterial color="#1B6FBF" transparent opacity={0.55} />
          </line>
        ))}
      </group>
    </group>
  );
}

export function Globe3D({ className = "" }: { className?: string }) {
  return (
    <div className={"absolute inset-0 " + className}>
      <Canvas camera={{ position: [0, 0, 5.5], fov: 50 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          <ambientLight intensity={0.6} />
          <Globe />
        </Suspense>
      </Canvas>
    </div>
  );
}
