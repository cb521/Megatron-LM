# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


"""Hopper M64 native-fragment softmax: no score shared-memory round trip."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Uint32, utils
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils

# Device functions used by this kernel.


@cute.jit
def swizzle_view(t):
    """Expose the shared-memory swizzle through the tensor pointer."""
    return cute.make_tensor(
        cute.recast_ptr(t.iterator, t.layout.inner, t.element_type), t.layout.outer
    )


@cute.jit
def vector_target(t, row, dim):
    # BF16 SW128 shared layout; dim starts an aligned group of eight values.
    """Address eight BF16 values in an SW128 shared-memory tile."""
    byte = cute.crd2idx((row, dim), t.layout.outer) * 2
    offset = (byte ^ ((byte >> 3) & 0x70)) // 2
    return cute.make_tensor((t.iterator + offset).align(16), cute.make_layout((8,)))


@cute.jit
def load_packed_query(
    q, sq, batch, group_index, tid, heads: cutlass.Constexpr, group: cutlass.Constexpr
):
    """Pack real query/head rows into M64 and zero unused rows."""
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), q.element_type, num_bits_per_copy=128)
    zero_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), q.element_type, num_bits_per_copy=128
    )
    zeros = cute.make_rmem_tensor((8,), q.element_type)
    zeros.fill(0)
    for i in cutlass.range_constexpr(16):
        item = (tid + i * 128) * 8
        row, d = item // 256, item % 256
        qi, head = group_index * group + row // heads, row % heads
        target = vector_target(sq, row, d)
        if row < group * heads and qi < q.shape[1]:
            off = cute.crd2idx((batch, qi, head, d), q.layout)
            source = cute.make_tensor((q.iterator + off).align(16), cute.make_layout((8,)))
            cute.copy(atom, source, target)
        else:
            cute.copy(zero_atom, zeros, target)


@cute.jit
def load_union_metadata(
    ids,
    bases,
    bits,
    counts,
    si,
    sm,
    fast,
    batch,
    group_index,
    tile,
    tid,
    group: cutlass.Constexpr,
    use_tma: cutlass.Constexpr,
):
    """Decode shared KV chunks and per-query membership into shared memory."""
    if tid < 64:
        block, lane = tile * 2 + tid // 32, tid % 32
        members, token = Int32(0), Int32(0)
        if block < counts[batch, group_index]:
            base = bases[batch, group_index, block]
            for g in cutlass.range_constexpr(group):
                word = bits[batch, group_index, block, g].to(Uint32)
                members |= ((word >> lane) & Uint32(1)).to(Int32) << g
            if base >= 0:
                token = base + lane
            elif members != 0:
                candidate = -base - 1
                chunks = cute.ceil_div(ids.shape[2], 32)
                source_q = group_index * group + candidate // chunks
                source_slot = (candidate % chunks) * 32 + lane
                token = ids[batch, source_q, source_slot]
        si[tid] = token
        sm[tid] = members
        first = cute.arch.shuffle_sync(token, 0)
        full = cute.arch.vote_all_sync((members != 0) & (token == first + lane) & (first % 32 == 0))
        if lane == 0:
            flag = Int32(full) if cutlass.const_expr(use_tma) else Int32(0)
            fast[tid // 32] = flag
    cute.arch.sync_threads()


@cute.jit
def zero_padding(
    out, lse, batch, group_index, tid, heads: cutlass.Constexpr, group: cutlass.Constexpr
):
    """Write zero output and negative-infinite LSE for padding heads."""
    if out.shape[2] > heads:
        count = group * (out.shape[2] - heads) * 256
        for i in cutlass.range(cute.ceil_div(count, 128)):
            slot = tid + i * 128
            if slot < count:
                d = slot % 256
                hp = (slot // 256) % (out.shape[2] - heads)
                qi = group_index * group + slot // (256 * (out.shape[2] - heads))
                if qi < out.shape[1]:
                    out[batch, qi, heads + hp, d] = cutlass.Float32(0.0).to(out.element_type)
                    if d == 0:
                        lse[batch, qi, heads + hp] = -float("inf")


@cute.jit
def load_payload(
    k,
    v,
    tk,
    tv,
    sk_raw,
    sv_raw,
    sk,
    sv,
    si,
    sm,
    fast,
    batch,
    tid,
    barrier,
    phase,
    atom_k: cute.CopyAtom,
    atom_v: cute.CopyAtom,
    use_tma: cutlass.Constexpr,
):
    """Stage K and V using TMA for contiguous chunks or vector gathers."""
    num_fast = (fast[0] & 1) + (fast[1] & 1)
    if cutlass.const_expr(use_tma):
        if tid < 32 and num_fast > 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barrier.iterator, num_fast * 32768)
            for piece in cutlass.range_constexpr(2):
                if (fast[piece] & 1) != 0:
                    token = si[piece * 32]
                    for col in cutlass.range_constexpr(4):
                        gk = cute.local_tile(tk[None, None, batch], (32, 64), (token // 32, col))
                        gv = cute.local_tile(tv[None, None, batch], (32, 64), (token // 32, col))
                        dk = cute.local_tile(sk, (32, 64), (piece, col))
                        dv = cute.local_tile(sv, (32, 64), (piece, col))
                        lk, _, _ = copy_utils.tma_get_copy_fn(
                            atom_k, 0, cute.make_layout(1), gk, dk, single_stage=True
                        )
                        lv, _, _ = copy_utils.tma_get_copy_fn(
                            atom_v, 0, cute.make_layout(1), gv, dv, single_stage=True
                        )
                        lk(tma_bar_ptr=barrier.iterator)
                        lv(tma_bar_ptr=barrier.iterator)
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), k.element_type, num_bits_per_copy=128)
    zero_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), k.element_type, num_bits_per_copy=128
    )
    zero_values = cute.make_rmem_tensor((8,), k.element_type)
    zero_values.fill(0)
    for piece in cutlass.range_constexpr(2):
        if (fast[piece] & 1) == 0:
            for item in cutlass.range_constexpr(8):
                index = (tid + item * 128) * 8
                row, d = piece * 32 + index // 256, index % 256
                dstk, dstv = (vector_target(sk_raw, row, d), vector_target(sv_raw, row, d))
                if sm[row] != 0:
                    off = cute.crd2idx((batch, si[row], d), k.layout)
                    gk = cute.make_tensor((k.iterator + off).align(16), cute.make_layout((8,)))
                    gv = cute.make_tensor((v.iterator + off).align(16), cute.make_layout((8,)))
                    cute.copy(atom, gk, dstk)
                    cute.copy(atom, gv, dstv)
                else:
                    # Invalid routes contribute zeros; each store covers eight BF16 values.
                    cute.copy(zero_atom, zero_values, dstk)
                    cute.copy(zero_atom, zero_values, dstv)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    if cutlass.const_expr(use_tma):
        if num_fast > 0:
            cute.arch.mbarrier_wait(barrier.iterator, phase)
            phase ^= 1
    cute.arch.sync_threads()
    cute.arch.fence_view_async_shared()
    return phase


@cute.jit
def store_pair(raw, row, col, x0, x1):
    """Store two BF16 probabilities together into an SW128 shared tile."""
    byte = cute.crd2idx((row, col), raw.layout.outer) * 2
    offset = (byte ^ ((byte >> 3) & 0x70)) // 2
    target = cute.make_tensor((raw.iterator + offset).align(4), cute.make_layout((2,)))
    values = cute.make_rmem_tensor((2,), raw.element_type)
    values[0] = x0.to(raw.element_type)
    values[1] = x1.to(raw.element_type)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), raw.element_type, num_bits_per_copy=32)
    cute.copy(atom, values, target)


# Attention kernel.


class PackedForward:
    """Hopper WGMMA attention over packed queries and their shared KV union."""

    def __init__(self, scale, heads=12, group=5, use_tma=True):
        self.scale_log2 = scale * 1.4426950408889634
        self.heads, self.group, self.use_tma = heads, group, use_tma

    @cute.jit
    def forward(self, q, k, v, ids, bases, bits, counts, out, lse, stream: cuda.CUstream):
        """Build layouts and launch one CTA per packed query group."""
        qk = sm90_utils.make_tiled_mma(q.element_type, "K", "K", 64)
        pv = sm90_utils.make_tiled_mma(q.element_type, "K", "MN", 256)
        layout = sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (32, 64))
        ak, tk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(k, [1, 2, 0]), layout, (32, 64)
        )
        av, tv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(v, [1, 2, 0]), layout, (32, 64)
        )
        self.kernel(q, k, v, ids, bases, bits, counts, out, lse, tk, tv, qk, pv, ak, av).launch(
            grid=(cute.ceil_div(q.shape[1], self.group), q.shape[0], 1),
            block=(128, 1, 1),
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        ids,
        bases,
        bits,
        counts,
        out,
        lse,
        tk,
        tv,
        mma_qk: cute.TiledMma,
        mma_pv: cute.TiledMma,
        atom_k: cute.CopyAtom,
        atom_v: cute.CopyAtom,
    ):
        """Compute packed sparse softmax attention and base-2 LSE."""
        tid, _, _ = cute.arch.thread_idx()
        gi, batch, _ = cute.arch.block_idx()
        alloc = utils.SmemAllocator()
        barrier = alloc.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=8)
        if tid < 32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier.iterator, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        phase = Int32(0)
        sq_raw = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        sk_raw = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        sv_raw = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        sp_raw = cute.make_tensor(
            sk_raw.iterator + 10240,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 64)),
        )
        sq, sk, sv, sp = (
            swizzle_view(sq_raw),
            swizzle_view(sk_raw),
            swizzle_view(sv_raw),
            swizzle_view(sp_raw),
        )
        si = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        sm = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        fast = alloc.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=8)
        cs = mma_qk.get_slice(tid).partition_C(cute.make_identity_tensor((64, 64)))
        co = mma_pv.get_slice(tid).partition_C(cute.make_identity_tensor((64, 256)))
        acc_s, fq, fk = sm90_utils.partition_fragment_ABC(
            mma_qk.get_slice(0), (64, 64, 256), sq, sk
        )
        acc_o, fp, fv = sm90_utils.partition_fragment_ABC(
            mma_pv.get_slice(0), (64, 256, 64), sp, layout_utils.transpose_view(sv)
        )
        acc_o.fill(0.0)
        # WGMMA M64 fragments: 4 contiguous threads share each row;
        # each thread owns two rows and two adjacent columns per N8 band.
        assert cute.size(acc_s) == 32
        running_max = cute.make_rmem_tensor((2,), Float32)
        denominator = cute.make_rmem_tensor((2,), Float32)
        alpha = cute.make_rmem_tensor((2,), Float32)
        running_max.fill(-float("inf"))
        denominator.fill(0.0)
        load_packed_query(q, sq_raw, batch, gi, tid, self.heads, self.group)
        cute.arch.cp_async_commit_group()
        for tile in cutlass.range(cute.ceil_div(counts[batch, gi], 2), unroll=1):
            load_union_metadata(
                ids,
                bases,
                bits,
                counts,
                si,
                sm,
                fast,
                batch,
                gi,
                tile,
                tid,
                self.group,
                self.use_tma,
            )
            phase = load_payload(
                k,
                v,
                tk,
                tv,
                sk_raw,
                sv_raw,
                sk,
                sv,
                si,
                sm,
                fast,
                batch,
                tid,
                barrier,
                phase,
                atom_k,
                atom_v,
                self.use_tma,
            )
            sm90_utils.gemm(mma_qk, acc_s, fq, fk, zero_init=True, wg_wait=0)
            for rr in cutlass.range_constexpr(2):
                row, _ = cs[rr * 2]
                qi = gi * self.group + row // self.heads
                member_bit = Int32(1) << (row // self.heads)
                row_valid = (row < self.group * self.heads) & (qi < q.shape[1])
                tile_max = Float32(-float("inf"))
                for cc in cutlass.range_constexpr(16):
                    i = cc % 2 + rr * 2 + (cc // 2) * 4
                    _, col = cs[i]
                    value = Float32(-float("inf"))
                    if row_valid and (sm[col] & member_bit) != 0:
                        value = acc_s[i] * self.scale_log2
                    acc_s[i] = value
                    tile_max = cute.arch.fmax(tile_max, value)
                for delta in cutlass.range_constexpr(2):
                    tile_max = cute.arch.fmax(
                        tile_max, cute.arch.shuffle_sync_bfly(tile_max, 1 << delta)
                    )
                next_max = cute.arch.fmax(running_max[rr], tile_max)
                scale = Float32(1.0)
                if next_max != -float("inf"):
                    scale = cute.math.exp2(running_max[rr] - next_max)
                total = Float32(0.0)
                for pair in cutlass.range_constexpr(8):
                    i = rr * 2 + pair * 4
                    _, col = cs[i]
                    p0, p1 = Float32(0.0), Float32(0.0)
                    if acc_s[i] != -float("inf"):
                        p0 = cute.math.exp2(acc_s[i] - next_max)
                    if acc_s[i + 1] != -float("inf"):
                        p1 = cute.math.exp2(acc_s[i + 1] - next_max)
                    store_pair(sp_raw, row, col, p0, p1)
                    total += p0
                    total += p1
                for delta in cutlass.range_constexpr(2):
                    total += cute.arch.shuffle_sync_bfly(total, 1 << delta)
                denominator[rr] = denominator[rr] * scale + total
                running_max[rr] = next_max
                alpha[rr] = scale
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(cute.size(acc_o)):
                acc_o[i] *= alpha[(i % 4) // 2]
            cute.arch.fence_view_async_shared()
            sm90_utils.gemm(mma_pv, acc_o, fp, fv, zero_init=False, wg_wait=0)
            cute.arch.sync_threads()
        cute.arch.cp_async_wait_group(0)
        if tid % 4 == 0:
            for rr in cutlass.range_constexpr(2):
                row, _ = cs[rr * 2]
                qi = gi * self.group + row // self.heads
                if row < self.group * self.heads and qi < q.shape[1]:
                    logsum = Float32(-float("inf"))
                    if denominator[rr] > 0.0:
                        logsum = running_max[rr] + cute.math.log2(denominator[rr])
                    lse[batch, qi, row % self.heads] = logsum
        for i in cutlass.range_constexpr(cute.size(acc_o)):
            r, d = co[i]
            query = gi * self.group + r // self.heads
            if r < self.group * self.heads and query < out.shape[1]:
                result = Float32(0.0)
                if denominator[(i % 4) // 2] > 0.0:
                    result = acc_o[i] / denominator[(i % 4) // 2]
                out[batch, query, r % self.heads, d] = result.to(out.element_type)
        zero_padding(out, lse, batch, gi, tid, self.heads, self.group)
