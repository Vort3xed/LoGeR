# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

try:
    import curope as _kernels # run `python setup.py install`
except ModuleNotFoundError:
    from . import curope as _kernels # run `python setup.py build_ext --inplace`


class cuRoPE2D_func (torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # tokens = tokens.clone() # uncomment this if inplace doesn't work
        _kernels.rope_2d( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.rope_2d( grad_res, positions, base, -F0 )
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0

    def forward(self, tokens, positions):
        # The kernel mutates the (B, seq, heads, dim) view in-place and asserts
        # contiguity. Tokens enter as (B, heads, seq, dim) which becomes
        # non-contiguous after transpose(1, 2), so we materialize a contiguous
        # copy, run the kernel on it, and return a re-transposed view (no
        # second copy). Callers always treat the result as fresh data.
        # NB: bypass the autograd.Function wrapper because we run in no_grad
        # for inference and the Function machinery adds noticeable per-call
        # python overhead (this RoPE is called ~50× per model forward).
        t_view = tokens.transpose(1, 2).contiguous()
        _kernels.rope_2d(t_view, positions, self.base, self.F0)
        return t_view.transpose(1, 2)