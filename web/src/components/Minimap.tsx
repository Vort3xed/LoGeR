// Top-down 2D minimap drawn to a canvas.
//
// Reads from the live trajectory buffers. We render at requestAnimationFrame
// rate but only repaint when len/resetTick changes — minimaps shouldn't dance
// while the user is still.

import { useEffect, useRef } from "react";

type Props = {
  anchors: Float32Array;
  anchorsLen: number;
  extraps: Float32Array;
  extrapsLen: number;
  resetTick: number;
  width?: number;
  height?: number;
};

export default function Minimap({
  anchors, anchorsLen, extraps, extrapsLen, resetTick,
  width = 220, height = 220,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr;
    cv.height = height * dpr;
    cv.style.width = `${width}px`;
    cv.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    let raf = 0;
    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(10,10,10,0.78)";
      ctx.fillRect(0, 0, width, height);

      // Bound the world by the active buffers (use X-Z, top-down — Y is up
      // in the model's convention).
      const useExtrap = extrapsLen > 0;
      const xs: number[] = [];
      const zs: number[] = [];
      for (let i = 0; i < anchorsLen; i++) {
        xs.push(anchors[i * 3]); zs.push(anchors[i * 3 + 2]);
      }
      for (let i = 0; i < extrapsLen; i++) {
        xs.push(extraps[i * 3]); zs.push(extraps[i * 3 + 2]);
      }
      if (xs.length === 0) {
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText("waiting for poses…", 14, 24);
        return;
      }
      const padding = 12;
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minZ = Math.min(...zs), maxZ = Math.max(...zs);
      const spanX = Math.max(maxX - minX, 1.0);
      const spanZ = Math.max(maxZ - minZ, 1.0);
      const span = Math.max(spanX, spanZ) * 1.15;
      const cx = (minX + maxX) / 2;
      const cz = (minZ + maxZ) / 2;
      const scale = (Math.min(width, height) - 2 * padding) / span;
      const toX = (x: number) => width / 2 + (x - cx) * scale;
      const toY = (z: number) => height / 2 - (z - cz) * scale; // flip so +Z goes up

      // Grid (1 m steps)
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      const step = 1.0;
      for (let g = Math.floor(minX / step) * step; g <= maxX + step; g += step) {
        ctx.beginPath();
        ctx.moveTo(toX(g), 0); ctx.lineTo(toX(g), height);
        ctx.stroke();
      }
      for (let g = Math.floor(minZ / step) * step; g <= maxZ + step; g += step) {
        ctx.beginPath();
        ctx.moveTo(0, toY(g)); ctx.lineTo(width, toY(g));
        ctx.stroke();
      }

      // Extrap (purple thin)
      if (extrapsLen > 1) {
        ctx.strokeStyle = "rgba(167,139,250,0.55)";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.moveTo(toX(extraps[0]), toY(extraps[2]));
        for (let i = 1; i < extrapsLen; i++) {
          ctx.lineTo(toX(extraps[i * 3]), toY(extraps[i * 3 + 2]));
        }
        ctx.stroke();
      }
      // Anchor (cyan thicker)
      if (anchorsLen > 1) {
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(toX(anchors[0]), toY(anchors[2]));
        for (let i = 1; i < anchorsLen; i++) {
          ctx.lineTo(toX(anchors[i * 3]), toY(anchors[i * 3 + 2]));
        }
        ctx.stroke();
      }

      // Current position
      const ci = useExtrap ? (extrapsLen - 1) * 3 : (anchorsLen - 1) * 3;
      const cxCur = useExtrap ? extraps[ci] : anchors[ci];
      const czCur = useExtrap ? extraps[ci + 2] : anchors[ci + 2];
      ctx.fillStyle = "#fbbf24";
      ctx.beginPath();
      ctx.arc(toX(cxCur), toY(czCur), 4, 0, Math.PI * 2);
      ctx.fill();

      // Scale label
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(`${span.toFixed(1)} m`, 8, height - 8);
    };

    const tick = () => { draw(); raf = requestAnimationFrame(tick); };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [anchors, anchorsLen, extraps, extrapsLen, resetTick, width, height]);

  return (
    <canvas
      ref={ref}
      className="rounded-xl border border-white/15 shadow-xl"
    />
  );
}
