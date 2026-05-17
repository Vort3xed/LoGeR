# LoGeR Setup Spec

## What is LoGeR?

LoGeR (Long-term Geometry-aware Reconstruction) is a monocular visual pose estimation model from Google DeepMind (Zhang et al., 2026, arXiv 2603.03269). It takes video frames as input and outputs per-frame 6DoF camera poses (position + orientation in 3D space). It is currently the most accurate published method for monocular camera pose estimation on indoor scenes.

**Paper:** https://arxiv.org/abs/2603.03269
**Repo:** https://github.com/Junyi42/LoGeR
**Checkpoints:** HuggingFace `Junyi42/LoGeR` (two variants: `LoGeR/latest.pt` and `LoGeR_star/latest.pt`)

## Why we're using it

We want to track a first responder walking through a building with a body cam/phone camera and reconstruct their 3D trajectory in real-time (or near-real-time with a sliding window approach). LoGeR is ideal because:
- Trained heavily on indoor data (ARKitScenes 14.6% of mix, ScanNet, etc.)
- Handles long sequences via overlapping windowed processing
- Produces metric-scale poses (not scale-ambiguous)
- Single monocular camera input — no stereo, no depth sensor, no IMU required
- KITTI ATE improved from 72.86m to 18.65m vs baseline (4x improvement)
- Best published rotation accuracy on Sintel (RPE-R 0.266 degrees)

## Architecture

LoGeR inherits from Pi3 (arXiv 2507.13347):
- **Backbone:** DINOv2 (frozen or fine-tuned) for per-frame feature extraction
- **Pose head:** 5-layer per-frame self-attention + MLP → 9D rotation (SVD orthogonalization) + 3D translation
- **Also outputs:** local pointmaps (per-frame depth/3D), confidence scores
- **Long sequence handling:** overlapping windows stitched via Umeyama Sim(3) alignment

Built on Pi3 + LaCT codebases. NOT built on DUSt3R/CroCo.

## Setup Requirements

- **GPU:** Needs a GPU. Model is ~1B params. Likely needs 16-24 GB VRAM minimum (A100, RTX 4090, RTX 3090 should work).
- **Python:** 3.9+
- **PyTorch:** 2.6+
- **Key dependencies:** torchvision, einops, roma, scipy, evo, accelerate
- **DINOv2** is vendored in the repo (no external download needed for backbone)
- Has `requirements.txt` and conda instructions

## How to run inference

```bash
# Clone
git clone https://github.com/Junyi42/LoGeR.git
cd LoGeR

# Install deps
pip install -r requirements.txt

# Download checkpoints from HuggingFace
# (check README for exact download instructions — likely huggingface_hub or wget)

# Run demo
bash demo_run.sh
# Or directly:
python demo_viser.py --input_path /path/to/images_or_video --config configs/loger.yaml --checkpoint path/to/latest.pt
```

The demo uses Viser for 3D visualization of the reconstructed trajectory + point cloud.

## Output format

Per-frame 4x4 camera-to-world matrices (SE(3)):
```
[[R(3x3) | t(3x1)]
 [0 0 0  |   1   ]]
```

Where R is the rotation matrix and t is the translation (camera position in world coordinates).

## Key parameters for our use case

- `window_size`: number of frames processed together (likely 24-48). Larger = more accurate, slower.
- `stride`: how many frames to advance between windows. Smaller = more overlap = smoother trajectory.
- `overlap`: frames shared between consecutive windows for stitching.

For near-real-time: use smaller window_size (16-24), process overlapping windows as new frames arrive.

## Important notes for the setup agent

1. The repo says "reimplementation" with "complete code and models will be released upon approval" — but the code IS complete (69 Python files) and checkpoints ARE available on HuggingFace. The disclaimer is a legal formality.

2. There are two checkpoint variants:
   - `LoGeR/latest.pt` — standard model
   - `LoGeR_star/latest.pt` — likely a stronger variant (check their README)

3. The evaluation pipeline uses the `evo` library (same one we use for D4RT eval). Pose metrics: ATE, RPE-T, RPE-R via Sim(3) alignment.

4. For our use case (first responder indoor tracking), test on:
   - A video of someone walking through a building
   - Check if the output trajectory is reasonable (straight corridors should look straight, turns should be ~90 degrees, etc.)
   - Metric scale should be approximately correct (a 10m hallway should measure ~10m in the output)

5. If you need to adapt this for streaming/real-time later:
   - The core model processes a fixed window of frames
   - For streaming: maintain a sliding window buffer, run the model on the latest N frames, extract the newest poses
   - Stitching between windows uses Umeyama alignment on overlapping frames
   - This is an engineering wrapper, not a model change

## Comparison with alternatives

| Method | Accuracy | Real-time? | Indoor trained? | Open source? |
|---|---|---|---|---|
| **LoGeR** | Best | No (batch) | Yes (heavy) | Yes (code + weights) |
| DROID-SLAM | Good | ~15 FPS | Somewhat | Yes |
| ORB-SLAM3 | Moderate | 30+ FPS | No | Yes |
| DPVO | Good | ~20 FPS | Somewhat | Yes |
| Our D4RT+PoseHead | Good (ATE 0.108) | ~45ms/clip | Yes (9 datasets) | Not released |

LoGeR is the accuracy leader. Real-time adaptation is our engineering task, not LoGeR's limitation.
