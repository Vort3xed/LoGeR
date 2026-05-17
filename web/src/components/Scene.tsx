// 3D scene: two trajectory lines (anchor + extrap), accumulated pointcloud.

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import * as THREE from "three";

import type { Pcd } from "../lib/trajectory";

type Props = {
  anchors: Float32Array;
  anchorsLen: number;
  extraps: Float32Array;
  extrapsLen: number;
  pcds: Pcd[];
  showAnchor: boolean;
  showExtrap: boolean;
  showPcd: boolean;
  resetTick: number;
  onRenderFps?: (fps: number) => void;
};

function PolyLine({
  buf, len, color, lineWidth,
}: { buf: Float32Array; len: number; color: string; lineWidth: number }) {
  const ref = useRef<THREE.Line>(null!);
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(buf, 3));
    g.setDrawRange(0, len);
    return g;
  }, [buf]);

  useFrame(() => {
    // BufferAttribute holds a reference to `buf` — when the array contents
    // change in place we just need to flag the attribute as dirty and
    // update the draw range. No realloc.
    geom.setDrawRange(0, len);
    (geom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  });

  return (
    <line ref={ref as any}>
      <primitive object={geom} attach="geometry" />
      <lineBasicMaterial color={color} linewidth={lineWidth} />
    </line>
  );
}

function PointsCloud({ pcds }: { pcds: Pcd[] }) {
  // Merge all pcds into a single buffer. Rebuild whenever the array of pcds
  // changes length (i.e. a new anchor's pcd arrived).
  const { positions, colors } = useMemo(() => {
    let total = 0;
    for (const p of pcds) total += p.xyz.length / 3;
    const positions = new Float32Array(total * 3);
    const colors = new Float32Array(total * 3);
    let off = 0;
    for (const p of pcds) {
      const n = p.xyz.length / 3;
      positions.set(p.xyz, off * 3);
      if (p.rgb) {
        for (let i = 0; i < n; i++) {
          colors[(off + i) * 3] = p.rgb[i * 3] / 255;
          colors[(off + i) * 3 + 1] = p.rgb[i * 3 + 1] / 255;
          colors[(off + i) * 3 + 2] = p.rgb[i * 3 + 2] / 255;
        }
      } else {
        for (let i = 0; i < n; i++) {
          colors[(off + i) * 3] = 0.75;
          colors[(off + i) * 3 + 1] = 0.75;
          colors[(off + i) * 3 + 2] = 0.75;
        }
      }
      off += n;
    }
    return { positions, colors };
  }, [pcds.length]);

  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return g;
  }, [positions, colors]);

  return (
    <points>
      <primitive object={geom} attach="geometry" />
      <pointsMaterial vertexColors size={0.012} sizeAttenuation />
    </points>
  );
}

function FpsProbe({ onRenderFps }: { onRenderFps?: (fps: number) => void }) {
  const last = useRef(performance.now());
  const acc = useRef({ frames: 0, lastReport: performance.now() });
  useFrame(() => {
    const now = performance.now();
    last.current = now;
    acc.current.frames += 1;
    if (now - acc.current.lastReport > 500) {
      const fps = (acc.current.frames * 1000) / (now - acc.current.lastReport);
      onRenderFps?.(fps);
      acc.current.frames = 0;
      acc.current.lastReport = now;
    }
  });
  return null;
}

function CameraFrustum({
  anchors, anchorsLen, extraps, extrapsLen,
}: { anchors: Float32Array; anchorsLen: number; extraps: Float32Array; extrapsLen: number }) {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame(() => {
    let x = 0, y = 0, z = 0;
    if (extrapsLen > 0) {
      const i = (extrapsLen - 1) * 3;
      x = extraps[i]; y = extraps[i + 1]; z = extraps[i + 2];
    } else if (anchorsLen > 0) {
      const i = (anchorsLen - 1) * 3;
      x = anchors[i]; y = anchors[i + 1]; z = anchors[i + 2];
    }
    ref.current?.position.set(x, y, z);
  });
  return (
    <mesh ref={ref}>
      <coneGeometry args={[0.04, 0.12, 12]} />
      <meshBasicMaterial color="#fbbf24" />
    </mesh>
  );
}

export default function Scene(props: Props) {
  return (
    <Canvas
      camera={{ position: [2.5, 2.5, 2.5], fov: 55, near: 0.01, far: 200 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#0a0a0a"]} />
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      <Grid
        infiniteGrid
        cellSize={0.5}
        sectionSize={2}
        cellColor="#444"
        sectionColor="#888"
        fadeDistance={30}
      />
      {props.showAnchor && props.anchorsLen > 1 && (
        <PolyLine
          key={`anchor-${props.resetTick}`}
          buf={props.anchors}
          len={props.anchorsLen}
          color="#22d3ee"
          lineWidth={2}
        />
      )}
      {props.showExtrap && props.extrapsLen > 1 && (
        <PolyLine
          key={`extrap-${props.resetTick}`}
          buf={props.extraps}
          len={props.extrapsLen}
          color="#a78bfa"
          lineWidth={1}
        />
      )}
      {props.showPcd && props.pcds.length > 0 && (
        <PointsCloud key={`pcd-${props.resetTick}-${props.pcds.length}`} pcds={props.pcds} />
      )}
      <CameraFrustum
        anchors={props.anchors}
        anchorsLen={props.anchorsLen}
        extraps={props.extraps}
        extrapsLen={props.extrapsLen}
      />
      <FpsProbe onRenderFps={props.onRenderFps} />
      <OrbitControls makeDefault dampingFactor={0.1} />
    </Canvas>
  );
}
