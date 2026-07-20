import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
torch._dynamo.config.compiled_autograd = True

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    update(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                        eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                    )
                futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_capped = CAPPED_HEADS[layer_idx]
        assert self.num_capped < num_heads
        # heterogeneous GQA: the capped (local) heads share KV head 0; full heads keep private KV
        self.num_kv_heads = num_heads - self.num_capped + (1 if self.num_capped else 0)
        hdim = num_heads * head_dim
        kv_hdim = self.num_kv_heads * head_dim
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # split into batched (q, o) and batched (k, v) so the K/V projections shrink with num_kv_heads
        # while Muon still orthogonalizes each projection separately
        self.qo_w = nn.Parameter(init_linear(torch.empty(2, hdim, dim)).bfloat16())
        self.qo_w.detach()[1].zero_() # out zero init suggested by @Grad62304977
        self.kv_w = nn.Parameter(init_linear(torch.empty(2, kv_hdim, dim)).bfloat16())
        self.rotary = Rotary(head_dim, max_seq_len)
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12
        self.qk_capture: list | None = None # when set to a list, forward appends (q, k) for attention-distance analysis
        self.q_sel_idx: Tensor | None = None # when also set, only these query positions of q are captured (keys stay full)

    def forward(self, x: Tensor, ve: Tensor | None, block_masks: tuple[BlockMask, BlockMask], lambdas: Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        # single fused QKV GEMM: cat the (tiny) weights rather than paying two GEMMs that each
        # re-read the (huge) activation tensor; the column count still shrinks with num_kv_heads
        qkv = F.linear(x, torch.cat([self.qo_w[0], self.kv_w.flatten(end_dim=1)]))
        qkv = qkv.view(B, T, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q = qkv[:, :, :self.num_heads]
        k, v = qkv[:, :, self.num_heads:].chunk(2, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        v = norm(v)
        if ve is not None:
            ve = ve.view(B, T, self.num_heads, self.head_dim)
            if self.num_capped: # the shared KV head takes the mean of the capped heads' value-embedding chunks
                ve = torch.cat([ve[:, :, :self.num_capped].mean(dim=2, keepdim=True), ve[:, :, self.num_capped:]], dim=2)
            v = lambdas[0] * v + lambdas[1] * ve # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = lambdas[0] * v
        attn_fn = flex_attention
        if self.qk_capture is not None: # analysis pass (runs outside the compiled model, see attn_distance_histograms)
            self.qk_capture.append((q if self.q_sel_idx is None else q[:, self.q_sel_idx], k))
            # eager flex_attention ignores the block mask's kv sparsity (the sliding window), so use a compiled one
            attn_fn = analysis_flex_attention
        full_bm, capped_bm = block_masks
        n = self.num_capped
        if n: # capped heads: MQA onto the shared KV head under the capped window; full heads: MHA under the scheduled window
            yc = attn_fn(q[:, :, :n].transpose(1, 2), k[:, :, :1].transpose(1, 2), v[:, :, :1].transpose(1, 2),
                         block_mask=capped_bm, scale=self.attn_scale, enable_gqa=True)
            yf = attn_fn(q[:, :, n:].transpose(1, 2), k[:, :, 1:].transpose(1, 2), v[:, :, 1:].transpose(1, 2),
                         block_mask=full_bm, scale=self.attn_scale)
            y = torch.cat([yc, yf], dim=1).transpose(1, 2)
        else:
            y = attn_fn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=full_bm, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qo_w[1])
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.proj_w)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, layer_idx) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_masks: tuple | None, lambdas: Tensor, sa_lambdas: Tensor):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_masks, sa_lambdas)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

# Long-short SWA by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper:
# these layers use the long (full-size) sliding window; all other attention layers use the short (half-size) one
LONG_WINDOW_LAYERS = (0, 4, 11, 15)
# Per-head sliding-window caps, informed by the attn_dist histograms (most heads are far more local
# than their window): in each layer, the first CAPPED_HEADS[i] of the 8 heads are restricted to
# HEAD_CAP_BLOCKS blocks for the whole run; the rest keep the full scheduled window. Which indices
# are capped is arbitrary - the mask breaks the symmetry and assigns the local roles to those heads.
# The capped (local) heads of a layer also SHARE ONE KV head (heterogeneous GQA/MQA): local heads do
# simple work, so they get one common key/value stream, shrinking the compute-bound K/V projections;
# full heads keep private KV. Layers 13/14 are uncapped - their histograms show no local specialists.
HEAD_CAP_BLOCKS = 2 # capped heads attend within 2*128 = 256 tokens (block-granular, so reach is 129-256 depending on position)
CAPPED_HEADS = (5, 6, 6, 6, 5, 5, 5, 0, 5, 5, 4, 4, 4, 0, 0, 2) # of 8 heads, per layer; layer 7 has no attention

def kv_head_map(layer_idx: int, num_heads: int) -> list[int]:
    """q-head -> kv-head index under the shared-KV grouping: capped heads all read kv head 0,
    full heads read their own private kv heads 1..num_heads-n."""
    n = CAPPED_HEADS[layer_idx]
    return [0] * n + list(range(1, num_heads - n + 1)) if n > 0 else list(range(num_heads))

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
        ]))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        def dense_to_ordered_asc(dense_blockmask: Tensor):
            # True blocks first, in ascending index order: for the q (backward) direction the
            # nearest in-window blocks are the lowest query-block indices
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = (~dense_blockmask).to(torch.int8).argsort(dim=-1, stable=True).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        blockmask_partial = blockmask_any & ~blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_partial)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        # q-direction (backward) base tables from the transposed block masks. Building these here and
        # window-clamping the counts below (mirroring the kv direction) skips from_kv_blocks'
        # _transpose_ordered, which re-materializes a dense mask + full argsort per (variant, head).
        partial_q_num_blocks, partial_q_indices = dense_to_ordered_asc(blockmask_partial.mT)
        full_q_num_blocks, full_q_indices = dense_to_ordered_asc(blockmask_all.mT)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            # nearest-w-blocks window via count clamping, applied identically in both directions so the
            # backward tables are exactly the transpose of the forward mask (verified vs from_kv_blocks)
            w = window_size_blocks
            return BlockMask(
                seq_lengths=(len(input_seq), len(input_seq)),
                kv_num_blocks=torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(w - full_kv_num_blocks, 1)),
                kv_indices=partial_kv_indices,
                full_kv_num_blocks=torch.clamp_max(full_kv_num_blocks, w - 1),
                full_kv_indices=full_kv_indices,
                q_num_blocks=torch.clamp_max(partial_q_num_blocks, torch.clamp_min(w - full_q_num_blocks, 1)),
                q_indices=partial_q_indices,
                full_q_num_blocks=torch.clamp_max(full_q_num_blocks, w - 1).clamp_min(0), # w=0 short masks: from_kv_blocks' transpose yields 0, not -1
                full_q_indices=full_q_indices,
                BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE),
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper.
        # Each attention layer gets a (full-window, capped-window) mask pair: the capped (shared-KV)
        # head group runs under the capped mask, the full heads under the scheduled one.
        long_full = build_bm(sliding_window_num_blocks)
        short_full = build_bm(sliding_window_num_blocks // 2)
        long_capped = build_bm(torch.clamp_max(sliding_window_num_blocks, HEAD_CAP_BLOCKS))
        short_capped = build_bm(torch.clamp_max(sliding_window_num_blocks // 2, HEAD_CAP_BLOCKS))
        return [None if block.attn is None else ((long_full, long_capped) if i in LONG_WINDOW_LAYERS else (short_full, short_capped))
                for i, block in enumerate(self.blocks)]

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        block_masks = self.create_blockmasks(input_seq, sliding_window_num_blocks) # per-layer (full, capped) mask pairs
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        skip_connections = []
        skip_map = {
            9: 6,
            10: 4,
            11: 2,
        }
        skip_weights = self.scalars[:len(self.blocks)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        for i in range(len(self.blocks)):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        # 16 chunks (up from 4) keeps the fp32 logits transient ~3GB instead of ~13GB, which the
        # uncompiled attn-viz pass at val_seq_len pays in full; equal chunks, so the loss is unchanged
        loss = 0
        for i in range(16):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(16)[i], self.lm_head_w.bfloat16()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(16)[i]) / 16
        return loss

# -----------------------------------------------------------------------------
# Attention distance analysis
# Measures two things separately:
#   1. where attention mass is allocated as a function of query-key distance d = i - j
#      (dense causal+document attention recomputed from the model's own q/k), and
#   2. how much masking out distant tokens changes the val loss (the window_sweep in the
#      training loop) - removal-based output error, closer to actual importance.

# The uncompiled flex_attention fallback applies only mask_mod (causal+document) and ignores
# the BlockMask's kv-block sparsity, i.e. the sliding window. So the analysis forward pass
# (which runs outside the compiled model) routes attention through this separately compiled
# flex_attention to reproduce the exact training-regime residual stream. Compiles lazily on first use.
# The GQA head-group split gives this function ~18 distinct shape signatures ((Hq, Hkv) per layer
# group, x histogram/val seq lens). Above the default recompile limit (8), dynamo silently falls
# back to UNCOMPILED flex, which materializes the full TxT score matrix and OOMs at these lengths.
torch._dynamo.config.recompile_limit = 40
analysis_flex_attention = torch.compile(flex_attention, dynamic=False)

@contextmanager
def capture_qk(attn_layers: list, q_sel_idx: Tensor | None = None):
    """Arm qk_capture (and optional query-row slicing via q_sel_idx) on the given layers,
    yield the per-layer capture lists that forward fills, and always disarm both fields."""
    for _, attn in attn_layers:
        attn.qk_capture = []
        attn.q_sel_idx = q_sel_idx
    try:
        yield [attn.qk_capture for _, attn in attn_layers]
    finally:
        for _, attn in attn_layers:
            attn.qk_capture = None
            attn.q_sel_idx = None

@torch.no_grad()
def attn_distance_histograms(model: GPT, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor, q_chunk=2048):
    """Per (attention layer, head), histogram over distance d = i - j of dense (causal +
    document masked) attention mass, summed over query tokens. The residual stream feeding
    each layer's q/k is the model's own, computed under its usual sliding-window masks; only
    the histogram itself uses dense attention, so cumulative mass C(w) answers "what fraction
    of dense attention would a sliding window of size w retain".
    Returns (mass: (L, H, T), count: (T,), layer_ids) where count[d] is the number of
    eligible (query, key) pairs at distance d."""
    attn_layers = [(i, block.attn) for i, block in enumerate(model.blocks) if block.attn is not None]
    with capture_qk(attn_layers) as captures:
        model(input_seq, target_seq, sliding_window_num_blocks)
    qks = []
    for (layer_id, _), capture in zip(attn_layers, captures):
        (q, k), = capture # each layer captured exactly once; q: (1, T, H, D), k: (1, T, num_kv_heads, D)
        kh = k[0].transpose(0, 1)[kv_head_map(layer_id, q.size(2))] # expand shared KV back to one row per q head
        qks.append((q[0].transpose(0, 1), kh)) # (H, T, D)
    T = input_seq.size(0)
    num_heads = qks[0][0].size(0)
    docs = (input_seq == 50256).cumsum(0)
    pos = torch.arange(T, device=input_seq.device)
    mass = torch.zeros(len(attn_layers), num_heads, T, device=input_seq.device)
    count = torch.zeros(T, device=input_seq.device)
    for i0 in range(0, T, q_chunk): # chunk over query tokens to bound memory
        d = pos[i0:i0 + q_chunk, None] - pos[None, :] # (Q, T) query-key distance
        valid = (d >= 0) & (docs[i0:i0 + q_chunk, None] == docs[None, :]) # causal + document mask
        idx = d.clamp_min(0).flatten() # masked pairs get prob 0 / count 0, so the clamped index is harmless
        count.scatter_add_(0, idx, valid.flatten().to(count.dtype))
        for li, ((_, attn), (qh, kh)) in enumerate(zip(attn_layers, qks)):
            logits = (qh[:, i0:i0 + q_chunk] @ kh.mT).float() * attn.attn_scale # (H, Q, T)
            probs = logits.masked_fill_(~valid, float("-inf")).softmax(-1) # (H, Q, T)
            for h in range(num_heads):
                mass[li, h].scatter_add_(0, idx, probs[h].flatten())
    return mass, count, [i for i, _ in attn_layers]

def print_attn_distance_summary(mass: Tensor, count: Tensor, layer_ids: list, step: int):
    mass, count = mass.double().cpu(), count.double().cpu()
    num_heads, T = mass.size(1), mass.size(2)
    edges = [0, 1] # log2 bins: [0], [1], [2,3], [4,7], ...
    while edges[-1] < T:
        edges.append(min(2 * edges[-1], T))
    bins = list(zip(edges[:-1], edges[1:]))
    labels = [f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}" for lo, hi in bins]

    def radius(c: Tensor, frac: float): # smallest w such that C(w) >= frac
        return int(torch.searchsorted(c, frac))

    def summarize(tag: str, m: Tensor, heads: int, console: bool):
        p = m / m.sum() # distribution of attention mass over distance, i.e. E_{i,h}
        c = p.cumsum(0) # C(w): fraction of dense attention retained by a window of size w
        cw = " ".join(f"C{w}:{c[min(w, T - 1)]:.4f}" for w in (128, 896, 1792))
        print0(f"attn_dist step:{step} {tag} r50:{radius(c, 0.5)} r90:{radius(c, 0.9)} r95:{radius(c, 0.95)} {cw}", console=console)
        print0(f"attn_dist step:{step} {tag} mass " + " ".join(
            f"[{lab}]:{p[lo:hi].sum():.4f}" for (lo, hi), lab in zip(bins, labels)), console=console)
        # per-token density: average probability that one eligible key token at this distance receives
        print0(f"attn_dist step:{step} {tag} density " + " ".join(
            f"[{lab}]:{m[lo:hi].sum() / (heads * count[lo:hi].sum()):.2e}"
            for (lo, hi), lab in zip(bins, labels) if count[lo:hi].sum() > 0), console=console)

    print0(f"attn_dist step:{step} eligible_pairs " + " ".join(
        f"[{lab}]:{int(count[lo:hi].sum())}" for (lo, hi), lab in zip(bins, labels)))
    for li, layer in enumerate(layer_ids):
        summarize(f"layer:{layer}", mass[li].sum(dim=0), num_heads, console=False)
        head_cum = (mass[li] / mass[li].sum(dim=-1, keepdim=True)).cumsum(dim=-1)
        print0(f"attn_dist step:{step} layer:{layer} head_r90 " + " ".join(
            f"h{h}:{radius(head_cum[h], 0.9)}" for h in range(num_heads)))
    summarize("all_layers", mass.sum(dim=(0, 1)), num_heads * len(layer_ids), console=True)

@torch.no_grad()
def attn_query_rows(model: GPT, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor, query_idx: Tensor):
    """Exact attention distributions (each sums to 1) that the given query tokens place over
    the keys behind them, recomputed from the model's own q/k under the full causal + document
    + block-wise sliding-window mask. Long-window layers use sliding_window_num_blocks, short-
    window layers half that (min 1 block), and the first CAPPED_HEADS[layer] heads are capped at
    HEAD_CAP_BLOCKS blocks, mirroring create_blockmasks. Returns
    (rows: (L, H, Q, 128*wb_long), layer_ids) where rows[l, h, qi, d] is the weight query
    query_idx[qi] places on the key d tokens back (0 beyond that head's window)."""
    attn_layers = [(i, block.attn) for i, block in enumerate(model.blocks) if block.attn is not None]
    with capture_qk(attn_layers, q_sel_idx=query_idx) as captures:
        model(input_seq, target_seq, sliding_window_num_blocks)
    T = input_seq.size(0)
    # CAUTION: the mask below hand-mirrors create_blockmasks (doc token 50256, 128-token blocks,
    # short window = long // 2 with min 1 block). Softmax renormalizes under whatever mask is used,
    # so if create_blockmasks changes and this drifts, rows still sum to 1 and look plausible —
    # update the two together.
    wb_long = int(sliding_window_num_blocks)
    d_max = 128 * wb_long # max window reach: query in block b sees keys down to block b - (wb - 1), i.e. distance <= 128*wb - 1
    docs = (input_seq == 50256).cumsum(0)
    pos = torch.arange(T, device=input_seq.device)
    d = query_idx[:, None] - pos[None, :] # (Q, T) how far behind each query each key sits
    causal_doc = (d >= 0) & (docs[query_idx][:, None] == docs[None, :])
    num_heads = captures[0][0][0].size(-2)
    rows = torch.zeros(len(attn_layers), num_heads, len(query_idx), d_max, device=input_seq.device)
    for li, ((layer_id, attn), capture) in enumerate(zip(attn_layers, captures)):
        (q_sel, k), = capture # each layer captured exactly once; q_sel: (1, Q, H, D), k: (1, T, num_kv_heads, D)
        wb = wb_long if layer_id in LONG_WINDOW_LAYERS else max(wb_long // 2, 1)
        # per-head windows: the first CAPPED_HEADS[layer_id] heads are capped at HEAD_CAP_BLOCKS blocks
        wb_head = torch.tensor([min(wb, HEAD_CAP_BLOCKS)] * CAPPED_HEADS[layer_id] + [wb] * (num_heads - CAPPED_HEADS[layer_id]),
                               device=input_seq.device)
        block_dist = query_idx[:, None] // 128 - pos[None, :] // 128 # (Q, T)
        allowed = causal_doc[None] & (block_dist[None] < wb_head[:, None, None]) # (H, Q, T)
        kh = k[0].transpose(0, 1)[kv_head_map(layer_id, num_heads)] # expand shared KV back to one row per q head
        logits = (q_sel[0].transpose(0, 1) @ kh.mT).float() * attn.attn_scale # (H, Q, T)
        probs = logits.masked_fill_(~allowed, float("-inf")).softmax(-1)
        for qi in range(len(query_idx)):
            sel = causal_doc[qi] & (d[qi] < d_max) # capped heads have exactly-0 probs beyond their window
            rows[li, :, qi, d[qi][sel]] = probs[:, qi, sel]
    return rows, [i for i, _ in attn_layers]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 64*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 550 # number of iterations to run
    cooldown_frac = 0.6 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
    # attention distance analysis
    attn_stats_seq_len = 64*1024 # tokens per attention-distance histogram pass (runs at each val step); 0 disables
    attn_viz = True # save middle/last-token attention rows averaged over the val loop at each val step (plot with plot_attn_rows.py)
    window_sweep_blocks = (1, 2, 4, 7, 14, 64, 2048) # window sizes (128-token blocks) for the end-of-run removal sweep; 2048 blocks = dense at 256K val seq len; () disables
    profile_step = 20 # if > 0, dump a kernel-level profiler table for that one training step. The step still
    # trains but is excluded from train_time, so step_avg is not comparable to unprofiled runs; diagnosis only
args = Hyperparameters()

run_id = int(os.environ.get("RUN_ID", 0))
# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
#assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
torch.manual_seed(12)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
if master_process:
    run_id_full = f"{run_id:03d}_{uuid.uuid4()}"
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id_full}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)
from torch._logging._internal import trace_structured # noqa: E402
import torch._inductor.codecache # noqa: E402
import torch._inductor.graph # noqa: E402
def _patched_trace_structured(name, metadata_fn=lambda: {}, **kwargs):
    if name == "inductor_output_code":
        print0(f'inductor_output_code: {metadata_fn().get("filename", "Unknown")}')
    trace_structured(name, metadata_fn, **kwargs)
torch._inductor.codecache.trace_structured = _patched_trace_structured
torch._inductor.graph.trace_structured = _patched_trace_structured

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)
embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
scalar_params = [model.scalars]
head_params: list[nn.Parameter] = [model.lm_head_w]
# sanity check
params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
optimized_parameters_set = {p for params in params_collections for p in params}
assert optimized_parameters_set == {*model.parameters()}
assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

# init the optimizer(s)
adam_param_groups = [dict(params=head_params, lr=1/320), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.015)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]
def opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for group in opt.param_groups for p in group["params"]]
opt2params = {opt: opt_params(opt) for opt in optimizers}
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        return (1 - x) / args.cooldown_frac

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

eager_model: GPT = model # uncompiled reference for the attention-distance analysis pass
model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = copy.deepcopy(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]))
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#        Training and validation       #
########################################

torch.cuda.reset_peak_memory_stats()
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
dist.barrier()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        viz_rows, viz_batches = None, 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
                # --------------- MIDDLE/LAST TOKEN ATTENTION ROWS ---
                if args.attn_viz: # off the training clock; every rank captures its shard so the average covers the whole val set
                    query_idx = torch.tensor([inputs.size(0) // 2, inputs.size(0) - 1], device=inputs.device)
                    rows, viz_layer_ids = attn_query_rows(eager_model, inputs, targets, get_window_size_blocks(step), query_idx)
                    viz_rows = rows if viz_rows is None else viz_rows + rows
                    viz_batches += 1
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        if viz_rows is not None:
            dist.all_reduce(viz_rows, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        # --------------- ATTENTION DISTANCE ANALYSIS --------
        if master_process and args.attn_stats_seq_len > 0: # runs off the training clock; no collectives, so master only
            n = args.attn_stats_seq_len # reuse the first n tokens of the last val batch
            mass, count, layer_ids = attn_distance_histograms(eager_model, inputs[:n], targets[:n], get_window_size_blocks(step))
            print_attn_distance_summary(mass, count, layer_ids, step)
            torch.save(dict(step=step, mass=mass.cpu(), count=count.cpu(), layers=layer_ids),
                       f"logs/{run_id_full}_attn_stats_step{step:06d}.pt")
        if master_process and viz_rows is not None:
            torch.save(dict(step=step, rows=(viz_rows / viz_batches).cpu(), layers=viz_layer_ids,
                            long_layers=list(LONG_WINDOW_LAYERS), window_blocks=int(get_window_size_blocks(step)),
                            capped_heads=list(CAPPED_HEADS), head_cap_blocks=HEAD_CAP_BLOCKS,
                            seq_len=args.val_seq_len, query_names=["middle", "last"],
                            num_batches=viz_batches * world_size),
                       f"logs/{run_id_full}_attn_rows_step{step:06d}.pt")
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if last_step:
        # --------------- ATTENTION WINDOW SWEEP -------------
        # Removal-based importance: re-evaluate val loss with restricted (or dense) attention
        # windows. The mass histograms say where attention goes; this says how much the loss
        # actually degrades when tokens beyond each distance are masked out.
        if args.window_sweep_blocks:
            model.eval()
            sweep_steps = max(1, val_steps // 4)
            with torch.no_grad():
                for w in args.window_sweep_blocks:
                    sweep_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
                    sweep_loss = 0
                    for _ in range(sweep_steps):
                        inputs, targets = next(sweep_loader)
                        sweep_loss += model(inputs, targets, torch.tensor(w, dtype=torch.int32, device="cuda"))
                    sweep_loss /= sweep_steps
                    del sweep_loader
                    dist.all_reduce(sweep_loss, op=dist.ReduceOp.AVG)
                    print0(f"window_sweep step:{step} window_blocks:{w} window_tokens:{128*w} val_loss:{sweep_loss:.6f} val_tokens:{sweep_steps*val_batch_size}", console=True)
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id_full}", exist_ok=True)
            torch.save(log, f"logs/{run_id_full}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    if args.profile_step and step == args.profile_step:
        # kernel-level Amdahl table for one full training step (fwd + bwd + grad reduce + optimizer),
        # off the clock. Same math as the normal path (incl. lr/momentum schedule), so training is
        # unaffected; only this step's wall time is excluded from train_time.
        dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        from torch.profiler import profile, ProfilerActivity # avoid top level import
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            model(inputs, targets, get_window_size_blocks(step)).backward()
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * get_lr(step)
            for group in optimizer2.param_groups:
                frac = min(step / 300, 1) # momentum warmup for muon
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        print0(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40), console=True)
        # kernel-partition view with input shapes: self-CUDA sums to total GPU time without nesting
        # double-counts, and the shapes separate Muon's NS matmuls from the model GEMMs
        print0(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=40), console=True)
        if master_process:
            trace_path = f"logs/{run_id_full}_trace_step{step:06d}.json"
            prof.export_chrome_trace(trace_path)
            print0(f"chrome trace saved to {trace_path} - open at https://ui.perfetto.dev", console=True)
        print0(f"step:{step+1}/{train_steps} profiled; step excluded from train_time", console=True)
        dist.barrier()
        t0 = time.perf_counter()
        continue
    model(inputs, targets, get_window_size_blocks(step)).backward()
    opt2futures = {
        opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
        for opt, params in opt2params.items()
    }
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        torch.futures.collect_all(opt2futures[opt]).wait()
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
