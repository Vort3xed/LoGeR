"""Measure how much accuracy is lost if we force LoGeR's global attention to
be CAUSAL (each frame attends only to itself and earlier frames) at inference,
even though the model was trained with bidirectional global attention.

If this loss is small, true incremental decode via KV-cache becomes viable.
If it's large, we have to back off to other optimizations.
"""

import sys, glob, inspect, yaml, math, torch, numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.amp as amp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loger.models.pi3 import Pi3
from loger.models.layers.attention import get_causal_block_mask


def load_model():
    with open("ckpts/LoGeR/original_config.yaml") as f:
        cfg = yaml.safe_load(f)["model"]
    sig = inspect.signature(Pi3.__init__)
    valid = {n for n, p in sig.parameters.items()
             if n not in {"self", "args", "kwargs"}
             and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    kwargs = {k: cfg[k] for k in cfg if k in valid}
    model = Pi3(**kwargs)
    state = torch.load("ckpts/LoGeR/latest.pt", map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval().to("cuda")


def build_frames(image_dir, n, pixel_limit=255_000):
    paths = natsorted(glob.glob(f"{image_dir}/*.png"))[:n]
    with Image.open(paths[0]) as im:
        W, H = im.size
    scale = math.sqrt(pixel_limit / (W * H))
    Wt, Ht = W * scale, H * scale
    k, m = round(Wt / 14), round(Ht / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > Wt / Ht: k -= 1
        else: m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    to_t = transforms.ToTensor()
    return [to_t(Image.open(p).convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS))
            for p in paths], (TW, TH)


def install_causal_global_attention(model: Pi3):
    """Monkey-patch the odd-indexed decoder blocks to apply a causal *float*
    bias on the (B, N*hw, C) global-attention call shape: 0 for allowed
    positions, -inf where token at position q (frame q//P) is not allowed to
    attend to token at kv (frame kv//P).

    We use a float mask (instead of a BlockMask) because attention.py's
    `is_floating_point(attn_mask)` check would otherwise blow up on BlockMask,
    and the bool-tensor path silently drops the mask under bf16 + flash attn.
    """
    mask_cache = {}

    def _get_float_causal_mask(P, N, device, dtype):
        """Return (M, M) float mask where mask[q,k] = 0 if q_frame >= k_frame, -inf else."""
        key = (P, N, device.index if device.type == "cuda" else -1, dtype)
        if key in mask_cache:
            return mask_cache[key]
        M = N * P
        idx = torch.arange(M, device=device)
        q_frame = (idx // P)[:, None]                    # (M, 1)
        k_frame = (idx // P)[None, :]                    # (1, M)
        allowed = q_frame >= k_frame                     # (M, M) bool
        mask = torch.zeros(M, M, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        mask_cache[key] = mask
        return mask

    for i, blk in enumerate(model.decoder):
        if i % 2 == 0:
            continue  # frame-attention layer — within-frame only

        orig_forward = blk.forward

        def patched_forward(x, xpos=None, attn_mask=None, _orig=orig_forward, _blk=blk):
            if attn_mask is None and getattr(_blk, "_causal_per_frame_tokens", None) is not None:
                P = _blk._causal_per_frame_tokens
                M = x.shape[1]
                N = M // P
                attn_mask = _get_float_causal_mask(P, N, x.device, x.dtype)
            return _orig(x, xpos=xpos, attn_mask=attn_mask)

        blk.forward = patched_forward

    orig_decode = model.decode

    def patched_decode(hidden, N, H, W, **kw):
        # tokens per frame in the global-attention reshape:
        #   patch tokens + pe (1) + register tokens (patch_start_idx counts both)
        full_per_frame = (H // model.patch_size) * (W // model.patch_size) + model.patch_start_idx
        for i, blk in enumerate(model.decoder):
            if i % 2 == 1:
                blk._causal_per_frame_tokens = full_per_frame
        try:
            return orig_decode(hidden, N, H, W, **kw)
        finally:
            for i, blk in enumerate(model.decoder):
                if i % 2 == 1:
                    blk._causal_per_frame_tokens = None

    model.decode = patched_decode
    return model


def umeyama_align(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s = src - mu_s; d = dst - mu_d
    H = s.T @ d / len(src)
    U, S, Vt = np.linalg.svd(H)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R = Vt.T @ Sgn @ U.T
    var_s = (s ** 2).sum() / len(src)
    scale = (S * np.diag(Sgn)).sum() / var_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def rot_err_deg(R_ref, R_test):
    errs = []
    for i in range(len(R_ref)):
        Re = R_ref[i].T @ R_test[i]
        c = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
        errs.append(math.degrees(math.acos(c)))
    return np.array(errs)


def main():
    print("[setup] loading bidirectional model ...", flush=True)
    model_bi = load_model()
    print("[setup] loading causal-patched model ...", flush=True)
    model_ca = load_model()
    install_causal_global_attention(model_ca)

    WIN, OV = 8, 3
    N = 16  # 16 frames -> 2 windows with overlap
    frames, (TW, TH) = build_frames("data/examples/office", N)
    print(f"[data] {N} frames @ {TW}x{TH}")
    x = torch.stack(frames).unsqueeze(0).to("cuda")

    print("\n[run] bidirectional forward ...", flush=True)
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out_bi = model_bi(x, window_size=WIN, overlap_size=OV)
    poses_bi = out_bi["camera_poses"][0].float().cpu().numpy()

    print("[run] causal-mask forward ...", flush=True)
    with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
        out_ca = model_ca(x, window_size=WIN, overlap_size=OV)
    poses_ca = out_ca["camera_poses"][0].float().cpu().numpy()

    # ----- Per-frame raw pose error (NO alignment) -----
    # LoGeR uses the first frame as canonical (identity) for both runs, so
    # rotations and translations are directly comparable. Translations may
    # carry small scale diffs because they're predicted by the metric head;
    # rotations are unitless and don't have a scale ambiguity.
    print("\n=== Per-frame raw pose error ===")
    print(f"{'frame':>5}  {'t_bi (m)':>22}  {'t-diff mm':>10}  {'rot-diff deg':>13}")
    t_diffs = []
    r_diffs = []
    for i in range(N):
        t_b = poses_bi[i, :3, 3]
        t_c = poses_ca[i, :3, 3]
        t_diff = float(np.linalg.norm(t_b - t_c))
        t_diffs.append(t_diff)
        Re = poses_bi[i, :3, :3].T @ poses_ca[i, :3, :3]
        cos = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2.0))
        rd = math.degrees(math.acos(cos))
        r_diffs.append(rd)
        marker = " *" if i in (0, WIN-1, WIN, N-1) else ""
        if i % 2 == 0 or marker:
            print(f"  {i:>3}  {str(np.round(t_b, 3)):>22}  "
                  f"{t_diff*1000:9.2f}  {rd:12.3f}{marker}")

    print(f"\n  trans  max: {max(t_diffs)*1000:.2f} mm   "
          f"mean: {sum(t_diffs)/len(t_diffs)*1000:.2f} mm   "
          f"RMSE: {math.sqrt(sum(t**2 for t in t_diffs)/len(t_diffs))*1000:.2f} mm")
    print(f"  rot    max: {max(r_diffs):.3f} deg   "
          f"mean: {sum(r_diffs)/len(r_diffs):.3f} deg   "
          f"RMSE: {math.sqrt(sum(r**2 for r in r_diffs)/len(r_diffs)):.3f} deg")

    # ----- Scale alignment (translation-only) -----
    print("\n=== Translation scale check (Umeyama over translations only) ===")
    scale_only, _, _ = umeyama_align(poses_ca[:, :3, 3], poses_bi[:, :3, 3])
    print(f"  scale factor (causal -> bidir): {scale_only:.4f}    (1.0 = perfect)")

    rot_rmse = math.sqrt(sum(r**2 for r in r_diffs)/len(r_diffs))
    rot_max = max(r_diffs)
    if rot_max < 1.0:
        verdict = "EXCELLENT: incremental decode highly viable"
    elif rot_max < 3.0:
        verdict = "OK: incremental decode viable, modest accuracy cost"
    elif rot_max < 10.0:
        verdict = "MARGINAL: usable only for prototyping"
    else:
        verdict = "BAD: causal-approx unusable without retraining"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
