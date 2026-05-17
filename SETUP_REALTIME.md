# Real-Time LoGeR Pipeline — Setup on a Fresh Machine

End-to-end instructions for reproducing the streaming + real-time pipeline on a new GPU machine. Tested on Ubuntu 24.04 + NVIDIA L40S (48 GB), CUDA driver 550.x. Should work on any Ampere or newer NVIDIA GPU (compute capability ≥ 8.0) with ≥ 16 GB VRAM.

## 0. Hardware & driver prerequisites

- NVIDIA GPU, compute capability ≥ 8.0 (Ampere / Ada / Hopper). Verified on L40S; A6000, RTX 4090, RTX 3090, A100 should all work.
- ~16 GB VRAM at window size 8 (our default). The full model is ~1 B params, ~3.6 GB on disk, ~8 GB in fp32 active memory.
- NVIDIA driver supporting CUDA 12.4 (driver ≥ 550). Check with `nvidia-smi` — the top-right "CUDA Version" must read 12.4 or higher. The driver's CUDA version only needs to be **≥** what PyTorch is built against; we install PyTorch 2.6 + cu124.
- ~15 GB free disk (two 4.7 GB checkpoints + conda env + repo).
- Miniconda or Anaconda installed.

## 1. Clone the repo

```bash
git clone https://github.com/Junyi42/LoGeR.git
cd LoGeR
```

If you have local changes from this machine that aren't pushed yet (e.g. `loger/streaming.py`, `loger/realtime_pipeline.py`, `loger/trajectory.py`, `loger/visualizer.py`, `bench/`, `models/curope/`), push them first and pull on the new machine. Easiest: `git add` the new files, commit, push, then clone on the target.

## 2. Create the conda environment

Follow the repo README verbatim — conda, not venv:

```bash
conda create -n loger python=3.11 cmake=3.14.0 -y
conda activate loger
pip install -r requirements.txt
```

This installs PyTorch 2.6.0 + cu124 binaries (torch's bundled `libcudart.so.12` lives in `site-packages/nvidia/cuda_runtime/lib/` — no system CUDA needed for *running* the model).

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.6.0+cu124 12.4 True <your-GPU-name>`

## 3. Download model checkpoints

Two variants, ~4.7 GB each:

```bash
mkdir -p ckpts/LoGeR ckpts/LoGeR_star
wget -O ckpts/LoGeR/latest.pt \
  "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR/latest.pt?download=true"
wget -O ckpts/LoGeR_star/latest.pt \
  "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR_star/latest.pt?download=true"
```

Our real-time pipeline uses `ckpts/LoGeR/latest.pt` by default. `LoGeR_star` is the stronger variant — slightly slower, slightly more accurate; swap it in if you want.

## 4. Build the cuRoPE2D CUDA kernel (recommended — ~5–8% inference speedup)

The model code does `from models.curope import cuRoPE2D` in `loger/models/layers/pos_embed.py`. If the import fails, it silently falls back to a pure-PyTorch RoPE. The CUDA kernel is fp32-equivalent — same numerical accuracy, just faster.

The kernel source comes from Naver's CroCo project (CC BY-NC-SA 4.0).

### 4a. Get the kernel source

```bash
# From the repo root
mkdir -p models
git clone --depth 1 --filter=blob:none --sparse https://github.com/naver/croco.git /tmp/croco_src
cd /tmp/croco_src && git sparse-checkout set models/curope && cd -
cp -r /tmp/croco_src/models/curope ./models/curope
rm -rf /tmp/croco_src
touch models/__init__.py
```

You should now have:
```
models/
├── __init__.py
└── curope/
    ├── __init__.py
    ├── curope.cpp
    ├── curope2d.py
    ├── kernels.cu
    └── setup.py
```

### 4b. Install CUDA dev tools matching PyTorch's CUDA version

PyTorch was built against CUDA 12.4, so the kernel must be built against 12.4 headers/libs to be ABI-compatible. **Do this inside the conda env** so it's isolated and easy to remove later:

```bash
conda install -n loger -c nvidia/label/cuda-12.4.0 \
  cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-nvrtc-dev=12.4 \
  libcublas-dev=12.4 cuda-cccl=12.4 -y
```

Verify nvcc:

```bash
nvcc --version  # should report release 12.4
```

### 4c. Build the kernel

```bash
cd models/curope
python setup.py build_ext --inplace
cd ../..
```

Expect a few warnings from `kernels.cu` about narrowing conversions — harmless. A single `.so` file will appear: `models/curope/curope.cpython-311-x86_64-linux-gnu.so`.

### 4d. Verify the kernel loads

```bash
python -c "from models.curope import cuRoPE2D; import torch; m = cuRoPE2D(64).cuda(); x = torch.randn(2, 4, 16, 64, device='cuda'); pos = torch.zeros(2, 16, 2, dtype=torch.long, device='cuda'); print('cuRoPE2D OK:', m(x, pos).shape)"
```

Expected: `cuRoPE2D OK: torch.Size([2, 4, 16, 64])`.

### 4e. (Optional) Free up the dev-toolkit disk after build

The compiled `.so` links against torch's bundled cudart, not the conda dev libs. Once it's built and verified, you can uninstall the dev toolkit to recover ~600 MB:

```bash
conda remove -n loger cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev libcublas-dev cuda-cccl -y
```

Re-running step 4b–4c is the only way to rebuild later, so only do this if you're confident.

## 5. Verify base-model inference

Quick smoke test on three synthetic frames:

```bash
python -c "
import torch
from loger.models.pi3 import Pi3
model = Pi3.from_pretrained('ckpts/LoGeR/latest.pt').cuda().eval()
imgs = torch.randn(1, 3, 3, 434, 574, device='cuda')   # B, T, C, H, W
with torch.no_grad():
    out = model.forward(imgs)
print('output keys:', list(out.keys()))
print('camera_poses:', out['camera_poses'].shape)
"
```

Expected: `output keys: [...]` including `camera_poses` with shape `torch.Size([1, 3, 4, 4])`.

## 6. Run the real-time pipeline

The real-time pipeline (added in `loger/realtime_pipeline.py`) decouples a fast foreground (returns an extrapolated pose in ~0.2 ms) from a background inference worker (runs the model at ~3.2 Hz). See module docstrings for full API.

Minimal usage on a folder of frames:

```python
import cv2, glob
from pathlib import Path
from loger.realtime_pipeline import RealtimePipeline

pipe = RealtimePipeline(
    checkpoint='ckpts/LoGeR/latest.pt',
    window_size=8,
    device='cuda',
)
pipe.start()

for path in sorted(glob.glob('path/to/frames/*.png')):
    img = cv2.imread(path)                  # BGR uint8 HxWx3
    pose, meta = pipe.submit_frame(img)
    print(meta['frame_idx'], pose[:3, 3])   # x, y, z (meters)

pipe.stop()
trajectory = pipe.trajectory.get_all_poses()  # list of 4x4 c2w matrices
```

`pose` is the latest extrapolated pose available at submit time (returned in <1 ms). Fresh model anchors land at ~3.2 Hz and the global trajectory stitches them with Sim(3) alignment.

## 7. (Optional) Run benchmarks

The `bench/` directory has scripts for measuring speedups, accuracy at different resolutions, drift over distance, etc. The two most useful sanity checks:

```bash
python bench/test_curope.py                # verify kernel correctness + speedup
python bench/bench_streaming.py            # streaming wrapper vs naive baseline
```

## 8. Performance expectations

Numbers on L40S, window=8, full resolution (574×434, `pixel_limit=255_000`):

| Path                                | Latency / call | Anchor FPS |
|-------------------------------------|---------------:|-----------:|
| Naive `model.forward()`             | ~437 ms        | 2.3 Hz     |
| StreamingLoGeR (encoder cache + compile) | ~340 ms   | 2.9 Hz     |
| + cuRoPE2D                          | ~310 ms        | 3.2 Hz     |

Foreground `submit_frame()` returns in ~0.2 ms regardless — so a 30 FPS camera sees 30 FPS pose updates, with 1-in-~10 of them being a fresh model anchor.

Lower-resolution paths get faster but **lose pose accuracy fast** — at 364×266 you pick up ~16° rotation RMSE and 33% scale drift. Don't drop resolution unless you've measured the accuracy cost on your data.

## 9. Known gotchas

- **CUDA toolkit version mismatch.** If you install `cuda-nvcc=12.6` against torch 2.6+cu124, the kernel may build but crash at runtime with ABI errors. Always match torch's CUDA version (`torch.version.cuda`).
- **Compute capability < 8.0.** The kernel compiles for all archs torch supports, but pre-Ampere GPUs (V100, T4) haven't been tested with this pipeline.
- **First call is slow.** `torch.compile` (used inside StreamingLoGeR) takes 10–30 s to trace + warm up on the first forward pass. Warm up by calling `submit_frame` with a few dummy frames before timing or using it under load.
- **The `models/` directory at the repo root is separate from `loger/models/`.** The first holds the curope kernel; the second holds the vendored model code. Don't conflate them.
- **DINOv2 weights** are loaded from the LoGeR checkpoint, not downloaded separately. No HuggingFace login needed beyond the checkpoint fetch.
