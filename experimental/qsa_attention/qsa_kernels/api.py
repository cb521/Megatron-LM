"""Prepared-tensor API: compilation/allocation are outside the measured call.

Shapes: Q/dO [B,Q,H,256], K/V [B,K,256], ids/valid [B,Q,S].
LSE uses log base 2. Backward accepts FP32 Delta = sum(O * dO).
H100 groups queries using one route-mask kernel, then one attention kernel.
B200 uses one swapAB attention kernel. dK/dV buffers are additive.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cutlass.cute as cute
import torch
from flash_attn.cute.cute_dsl_utils import to_cute_tensor


@dataclass(frozen=True)
class PreparedAttention:
    """A plan owns fixed input/output buffers and runs on the current stream.

    run() rebuilds H100 route masks on each call. core() reuses the last masks:
    call preprocess() first if ids or valid changed. Buffers cannot be used by
    overlapping calls on different streams. Backward adds into dK/dV on every
    call; clear them explicitly when a fresh gradient is required.
    """

    run: Callable
    core: Callable
    preprocess: Callable
    output: tuple
    metadata: tuple
    kernel: object
    architecture: str
    heads: int
    group: int
    kernel_count: int

    def __call__(self) -> tuple[torch.Tensor, ...]:
        """Run the plan on the current CUDA stream."""
        return self.run()


def _tensor(t, name, dtype, shape=None, device=None):
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dtype != dtype or not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtype}")
    if device is not None and t.device != device:
        raise ValueError(f"{name} must be on {device}")
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} shape must be {tuple(shape)}, got {tuple(t.shape)}")
    if t.data_ptr() % 16:
        raise ValueError(f"{name} must have 16-byte aligned storage")


def _inputs(q, k, v, ids, valid, heads, group, scale):
    _tensor(q, "q", torch.bfloat16)
    if q.ndim != 4 or q.shape[-1] != 256 or min(q.shape) <= 0:
        raise ValueError("q must have shape [B,Q,H,256] with nonzero dimensions")
    batch, queries, padded_heads, _ = q.shape
    heads = padded_heads if heads is None else heads
    if isinstance(heads, bool) or not isinstance(heads, int) or not 1 <= heads <= padded_heads:
        raise ValueError("heads must be a positive integer no greater than q.shape[2]")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    _tensor(k, "k", torch.bfloat16, device=q.device)
    if k.ndim != 3 or k.shape[0] != batch or k.shape[-1] != 256 or k.shape[1] <= 0:
        raise ValueError("k must have shape [B,K,256] with K > 0")
    _tensor(v, "v", torch.bfloat16, k.shape, q.device)
    _tensor(ids, "ids", torch.int32, device=q.device)
    if ids.ndim != 3 or tuple(ids.shape[:2]) != (batch, queries) or ids.shape[-1] <= 0:
        raise ValueError("ids must have shape [B,Q,S] with S > 0")
    _tensor(valid, "valid", torch.int32, ids.shape, q.device)
    if group is not None and (isinstance(group, bool) or not isinstance(group, int)):
        raise ValueError("group must be an integer")
    arch = torch.cuda.get_device_capability(q.device)
    if arch == (9, 0):
        if group is None:
            group = next((g for g in (5, 4, 2, 1) if g * heads <= 64), None)
        if group not in (1, 2, 4, 5) or group * heads > 64:
            raise ValueError("H100 requires group in {1,2,4,5} and group*heads <= 64")
        capacity = max(128, 1 << (group * ((ids.shape[-1] + 31) // 32) - 1).bit_length())
        metadata_elements = batch * ((queries + group - 1) // group) * capacity * group
        if ids.numel() > 2**31 - 1 or metadata_elements > 2**31 - 1 or batch > 65535:
            raise ValueError("H100 routes exceed the supported int32 indexing/grid range")
        if group * ((ids.shape[-1] + 31) // 32) > 1024:
            raise ValueError("H100 route capacity exceeds 1024 chunks per query group")
        arch_name = "h100"
    elif arch == (10, 0):
        if padded_heads % 16:
            raise ValueError("B200 requires padded Q heads to be a multiple of 16")
        if group not in (None, 1):
            raise ValueError("B200 swapAB operates on one query per CTA")
        group = 1
        arch_name = "b200"
    else:
        raise ValueError(
            f"unsupported device capability {arch}; expected H100 (9,0) or B200 (10,0)"
        )
    # This is a preparation check, outside run()/core() and outside CUDA graphs.
    # The caller must retain this invariant after any subsequent route update.
    if bool(((valid != 0) & ((ids < 0) | (ids >= k.shape[1]))).any().item()):
        raise ValueError("every valid route ID must be in [0,K)")
    return arch_name, heads, group


def _compile(entry, args, dump_dir):
    options = "--enable-tvm-ffi"
    if dump_dir is not None:
        dest = Path(dump_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        if any(c.isspace() for c in str(dest)):
            raise ValueError("dump_dir must not contain whitespace")
        options += " --keep-ptx --keep-cubin --dump-dir=" + str(dest)
    return cute.compile(
        entry,
        *(to_cute_tensor(x) for x in args),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=options,
    )


def prepare_forward(
    args: tuple[torch.Tensor, ...],
    *,
    heads: int | None = None,
    group: int | None = None,
    scale: float = 0.0625,
    use_tma: bool = True,
    dump_dir: str | Path | None = None,
) -> PreparedAttention:
    """Compile a forward plan. ids may contain any value where valid == 0.

    On H100 the first `heads` entries of each padded Q row are evaluated; the
    remaining output entries are zero with LSE=-inf. On B200 all padded heads
    are evaluated; consumers slice the first `heads` entries.
    """
    if len(args) != 5:
        raise ValueError("forward expects (q,k,v,ids,valid)")
    q, k, v, ids, valid = args
    arch, heads, group = _inputs(q, k, v, ids, valid, heads, group, scale)
    with torch.cuda.device(q.device):
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        meta = ()

        def preprocess():
            return None

        if arch == "h100":
            from .forward_h100 import PackedForward
            from .metadata import metadata

            meta, preprocess = metadata(ids, valid, k.shape[1], group)
            impl = PackedForward(scale, heads, group, use_tma)
            core_args = (q, k, v, ids, *meta, output, lse)
        else:
            from .forward_b200 import TokenForward

            impl = TokenForward(scale, use_tma)
            core_args = (*args, output, lse)
        kernel = _compile(impl.forward, core_args, dump_dir)

    def core():
        kernel(*core_args)
        return output, lse

    def run():
        preprocess()
        return core()

    return PreparedAttention(
        run, core, preprocess, (output, lse), meta, kernel, arch, heads, group, 2 if meta else 1
    )


def prepare_backward(
    args: tuple[torch.Tensor, ...],
    *,
    heads: int | None = None,
    group: int | None = None,
    scale: float = 0.0625,
    use_tma: bool = True,
    dk: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    dump_dir: str | Path | None = None,
) -> PreparedAttention:
    """Compile backward with caller-owned log2 LSE and FP32 Delta.

    dQ is overwritten; FP32 dK/dV are accumulated. Omitted buffers start at zero.
    Computing Delta and clearing gradient buffers are separate caller operations.
    """
    if len(args) != 8:
        raise ValueError("backward expects (q,k,v,do,ids,valid,lse,delta)")
    q, k, v, do, ids, valid, lse, delta = args
    arch, heads, group = _inputs(q, k, v, ids, valid, heads, group, scale)
    _tensor(do, "do", torch.bfloat16, q.shape, q.device)
    _tensor(lse, "lse", torch.float32, q.shape[:-1], q.device)
    _tensor(delta, "delta", torch.float32, q.shape[:-1], q.device)
    with torch.cuda.device(q.device):
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32) if dk is None else dk
        dv = torch.zeros_like(v, dtype=torch.float32) if dv is None else dv
        _tensor(dk, "dk", torch.float32, k.shape, q.device)
        _tensor(dv, "dv", torch.float32, v.shape, q.device)
        if (
            dk.data_ptr() < dv.data_ptr() + dv.numel() * dv.element_size()
            and dv.data_ptr() < dk.data_ptr() + dk.numel() * dk.element_size()
        ):
            raise ValueError("dk and dv must not overlap")
        tensors = [q, k, v, do, ids, valid, lse, delta, dk, dv]
        for out in (dk, dv):
            start = out.data_ptr()
            end = start + out.numel() * out.element_size()
            for other in tensors:
                if other is out:
                    continue
                if (
                    start < other.data_ptr() + other.numel() * other.element_size()
                    and other.data_ptr() < end
                ):
                    raise ValueError(
                        "gradient buffers must not overlap each other or input buffers"
                    )
        meta = ()

        def preprocess():
            return None

        if arch == "h100":
            from .backward_h100 import PackedBackward
            from .metadata import metadata

            meta, preprocess = metadata(ids, valid, k.shape[1], group)
            impl = PackedBackward(scale, heads, group, use_tma)
            core_args = (q, k, v, do, ids, *meta, lse, delta, dq, dk, dv)
        else:
            from .backward_b200 import TokenBackward

            impl = TokenBackward(scale, use_tma)
            core_args = (*args, dq, dk, dv)
        kernel = _compile(impl.backward, core_args, dump_dir)

    def core():
        kernel(*core_args)
        return dq, dk, dv

    def run():
        preprocess()
        return core()

    return PreparedAttention(
        run, core, preprocess, (dq, dk, dv), meta, kernel, arch, heads, group, 2 if meta else 1
    )
