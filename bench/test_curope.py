"""Verify cuRoPE2D output matches the pure-PyTorch fallback, then benchmark.

Loads the model (which now picks up cuRoPE2D automatically — the import in
loger/models/layers/pos_embed.py finds it). Compares one forward pass against
a manual PyTorch RoPE on the same inputs to confirm parity, then measures the
inference latency improvement.
"""

import sys, time, statistics, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Import the python RoPE2D directly for parity comparison
from loger.models.layers.pos_embed import RoPE2D as CurrentRoPE2D


class PyRoPE2D(torch.nn.Module):
    """Verbatim copy of the pure-PyTorch RoPE2D from pos_embed.py (the fallback)."""
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq; self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        key = (D, seq_len, device, dtype)
        if key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            self.cache[key] = (freqs.cos(), freqs.sin())
        return self.cache[key]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2
        cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)


def main():
    device = "cuda"
    print(f"Current RoPE2D class: {CurrentRoPE2D}")
    is_cuda_kernel = "curope" in str(CurrentRoPE2D)
    print(f"Using CUDA kernel: {is_cuda_kernel}\n")

    # Same shapes as in real LoGeR forward — use FP32 for clean parity comparison
    B, num_heads, num_tokens, dim_per_head = 1 * 8, 16, 1271, 64
    tokens = torch.randn(B, num_heads, num_tokens, dim_per_head, device=device, dtype=torch.float32)
    # 2D positions: (B, num_tokens, 2)
    # Use a 41x31 grid layout matching 574x434 / 14
    from loger.models.layers.pos_embed import PositionGetter
    pg = PositionGetter()
    pos = pg(B, 31, 41, device).to(device)
    print(f"tokens: {tuple(tokens.shape)}  pos: {tuple(pos.shape)}\n")

    # Parity check — copy tokens first since the CUDA kernel mutates in-place
    print("=== Parity check (current vs pure-PyTorch fallback) ===")
    cu_layer = CurrentRoPE2D(freq=100.0).to(device)
    py_layer = PyRoPE2D(freq=100.0).to(device)
    with torch.no_grad():
        out_py = py_layer(tokens.clone(), pos)     # py first, doesn't mutate
        out_cu = cu_layer(tokens.clone(), pos)     # cu on a FRESH copy
    abs_diff = (out_cu.float() - out_py.float()).abs()
    rel_diff = abs_diff / (out_py.float().abs() + 1e-6)
    print(f"  abs diff: mean={abs_diff.mean().item():.2e}  max={abs_diff.max().item():.2e}")
    print(f"  rel diff: mean={rel_diff.mean().item():.2e}  max={rel_diff.max().item():.2e}")
    if abs_diff.max().item() < 1e-2:
        print(f"  -> outputs match within bf16 numerical tolerance")
    else:
        print(f"  -> outputs differ noticeably; investigate")

    # Benchmark
    def bench(layer, n=200, warmup=20):
        # Use a fresh tensor each call so the CUDA in-place mutation doesn't
        # accumulate angle drift over runs.
        for _ in range(warmup):
            _ = layer(tokens.clone(), pos)
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            x = tokens.clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = layer(x, pos)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return times

    print("\n=== Benchmark (single RoPE2D call, 200 runs) ===")
    print("  Pure PyTorch:", end=" ", flush=True)
    py_times = bench(py_layer)
    print(f"mean={statistics.mean(py_times)*1e6:.1f}us  "
          f"median={statistics.median(py_times)*1e6:.1f}us  "
          f"min={min(py_times)*1e6:.1f}us")
    print("  cuRoPE2D    :", end=" ", flush=True)
    cu_times = bench(cu_layer)
    print(f"mean={statistics.mean(cu_times)*1e6:.1f}us  "
          f"median={statistics.median(cu_times)*1e6:.1f}us  "
          f"min={min(cu_times)*1e6:.1f}us")
    speedup = statistics.mean(py_times) / statistics.mean(cu_times)
    print(f"\n  speedup: {speedup:.2f}x  ({(speedup-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
