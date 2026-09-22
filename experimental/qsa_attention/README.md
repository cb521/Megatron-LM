# QSA attention kernels for H100 and B200

Standalone BF16, D=256 forward/backward kernels for prepared sparse GQA inputs.
The package selects the implementation from the CUDA device:

| GPU | Implementation |
| --- | --- |
| H100 | Pack up to 5 Q tokens per CTA, compact shared KV routes, then use WGMMA. |
| B200 | One Q token per CTA/head group, with swapAB and `tcgen05.mma`. |

H100 runs route preparation before each forward/backward attention kernel.
B200 needs no separate route kernel. In B200 backward, QK, dP and dQ use
swapAB; dK and dV use their usual orientation.

Each attention kernel and its device functions live together in one file:
`forward_h100.py`, `backward_h100.py`, `forward_b200.py`, `backward_b200.py`.
`metadata.cu` contains H100 route preparation; `api.py` provides the common call interface.

## Install

Use Python 3.12 and a CUDA development container with a C++ compiler and `nvcc`.
From this directory, in an isolated environment:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install .
# Install the compiler libraries last; choose the target GPU.
python -m pip install --force-reinstall --no-deps nvidia-cutlass-dsl-libs-cu13==4.5.2  # H100
# python -m pip install --force-reinstall --no-deps nvidia-cutlass-dsl-libs-base==4.5.2  # B200
python -m qsa_kernels.validation
```

Validated versions: Torch 2.11.0+cu130, CUTLASS DSL 4.5.2, quack-kernels 0.5.0,
flash-attn-4 4.0.0b17 and cuda-python 13.2.0. The compiler libraries overlap:
the last-installed package determines the compiler (H100: 13.1; B200: 12.9).
H100 compiles the included `metadata.cu` extension on first preparation.

## Call

`Bp = batch_size * num_kv_heads`. Fold KV heads into the batch dimension,
group their Q heads, and pad each group to 16 heads for the measured cases.

| Tensor | Contiguous shape | Type |
| --- | --- | --- |
| Q, output, dO, dQ | `[Bp, Q_length, padded_heads, 256]` | BF16 |
| K, V | `[Bp, KV_length, 256]` | BF16 |
| Selected token IDs, valid flags | `[Bp, Q_length, slots]` | int32 |
| LSE, Delta | `[Bp, Q_length, padded_heads]` | FP32 |
| dK, dV | `[Bp, KV_length, 256]` | FP32 |

```python
from qsa_kernels import prepare_backward, prepare_forward

# heads is the actual Q-head count per KV head: 12 for 1030, 3 for 2064.
fwd = prepare_forward((q, k, v, ids, valid), heads=heads)
out, lse = fwd.run()

# dO must be zero in padding heads.
delta = (out.float() * do.float()).sum(-1).contiguous()
bwd = prepare_backward((q, k, v, do, ids, valid, lse, delta), heads=heads)
bwd.output[1].zero_()
bwd.output[2].zero_()
dq, dk, dv = bwd.run()
```

- `prepare_*` allocates and compiles; `run()` reuses fixed tensor addresses on
  the current stream. Update contents in place; prepare again if storage or
  shapes change. Each plan owns buffers and cannot run concurrently on streams.
- `valid != 0` alone defines visibility. Valid IDs must be in range. Apply
  causality when building routes; the kernel does not add another causal mask.
  Repeated IDs count once per occurrence. Invalid IDs may be out of range.
- LSE uses **base 2**. Pass the matching forward LSE to backward.
  H100 zeroes padding outputs and uses `-inf` padding LSE; B200 computes padding
  heads too. Use only actual heads, with zero dO/Delta in padding heads.
- Backward overwrites dQ and **adds into dK/dV**. Clear dK/dV before each fresh
  backward, including graph replays. Cast FP32 dK/dV only at the caller boundary.
- P/dS use BF16 and matrix accumulators use FP32. Floating-point atomics make
  dK/dV non-bitwise-deterministic; tiny cancellation gradients may retain residue.
- H100 supports `group` 1/2/4/5 with `group * heads <= 64` and
  `group * ceil(slots / 32) <= 1024`. B200 requires padded heads divisible by 16.
  H100 preserves duplicate routes through a fallback in the same metadata kernel.

This is the prepared attention-kernel boundary. Connect it inside the existing
TileLang wrapper after layout conversion and before restoring the model layout.
Full-model training integration is outside the measurements below.

## Performance against TileLang

Batch=1, BF16, D=256, Q length=KV length. Routes select blocks of 4 tokens;
budgets are 1024 for length 1030 and 2048 for length 2064, plus the causal tail.
Values below are **forward / backward**; latency is milliseconds.

| GPU | Length | Q/KV heads | TileLang ms | This package ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 | 1030 | 24/2 | 0.554 / 1.359 | 0.207 / 0.473 | 2.68x / 2.87x |
| H100 | 2064 | 3/1 | 1.037 / 2.639 | 0.415 / 0.860 | 2.50x / 3.07x |
| B200 | 1030 | 24/2 | 0.446 / 1.216 | 0.188 / 0.678 | 2.38x / 1.79x |
| B200 | 2064 | 3/1 | 0.891 / 2.356 | 0.327 / 1.273 | 2.72x / 1.85x |

| GPU | Length | TileLang effective TFLOPS | This package effective TFLOPS |
| --- | ---: | ---: | ---: |
| H100 | 1030 | 23.6 / 24.0 | 63.1 / 68.9 |
| H100 | 2064 | 6.3 / 6.2 | 15.8 / 19.0 |
| B200 | 1030 | 29.3 / 26.8 | 69.5 / 48.1 |
| B200 | 2064 | 7.3 / 6.9 | 20.0 / 12.9 |

These are archived paired measurements of this implementation: H100 on
2026-09-21, B200 on 2026-09-20 (unchanged by the H100 route update), using
40 CUDA-graph samples and identical synthetic inputs per comparison.
H100 includes route preparation; both backwards include dK/dV zeroing.
Delta, compilation, allocations, layout conversion and route selection are
excluded. Effective FLOPs count actual heads and valid pairs: 4D forward,
10D backward. These are single-GPU attention times, not full training times.

Baseline: AllenFeiZZ's [QSA implementation at a904109](https://github.com/AllenFeiZZ/Megatron-LM/tree/a904109c3a67a0c0535262ce4eb02bd83d933d6b),
with the two backward shared-memory barriers in [tilelang-sync.patch](tilelang-sync.patch).
TileLang 0.1.8 forward uses the faster tested 128-thread/1-stage setting;
backward retains its original settings. Both use the same prepared inputs.
Apply the patch to that commit with `git apply --unidiff-zero /path/to/tilelang-sync.patch`.
