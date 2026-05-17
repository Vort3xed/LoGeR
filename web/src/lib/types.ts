// Wire types matching server/realtime_server.py protocol.

export type FpsBlock = {
  anchor: number;
  extrap: number;
  inference_ms: number;
};

export type PoseMsg = {
  type: "pose";
  kind: "anchor" | "extrap";
  idx: number;
  t: number;
  pose: number[]; // 16 floats, c2w row-major
  fps: FpsBlock;
};

export type PcdMsg = {
  type: "pcd";
  idx: number;
  n: number;
  xyz: number[];        // 3n floats, world meters
  rgb: number[] | null; // 3n ints 0-255
};

export type ResetMsg = { type: "reset" };
export type PingMsg = { type: "ping" | "pong" };

export type DownlinkMsg = PoseMsg | PcdMsg | ResetMsg | PingMsg;

export type SavedRunSummary = {
  file: string;
  id: string;
  name: string;
  started_at: number;
  ended_at: number | null;
  anchors: number;
  pcds: number;
};
