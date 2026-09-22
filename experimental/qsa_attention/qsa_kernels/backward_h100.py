# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


"""M64 packed backward: stream V in halves to fit two CTAs per SM.

P/dS reuse V after dP. Register shuffles assemble contiguous groups of four
gradients for direct FP32 vector atomics; no shared gradient exchange is used.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Uint32, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils

# Device functions used by this kernel.


@dsl_user_op
def atomic_add4(ptr, x0, x1, x2, x3, *, loc=None, ip=None):
    """Atomically add four contiguous FP32 gradient values."""
    llvm.inline_asm(
        None,
        [ptr.llvm_ptr, x0.ir_value(), x1.ir_value(), x2.ir_value(), x3.ir_value()],
        "red.relaxed.gpu.global.add.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def swizzle_view(t):
    """Expose the shared-memory swizzle through the tensor pointer."""
    return cute.make_tensor(
        cute.recast_ptr(t.iterator, t.layout.inner, t.element_type), t.layout.outer
    )


@cute.jit
def contiguous_four(a0, a1, b0, b1, tid):
    # Transpose lane bits with the two N8 bands. Shuffles move FP32 bits;
    # this does not add, round or reorder the accumulated gradient values.
    """Redistribute four FP32 values per lane without changing their bits."""
    for bit in cutlass.range_constexpr(2):
        mask = 2 if bit == 0 else 1
        high = (tid & mask) != 0
        t0 = a0 if high else b0
        t1 = a1 if high else b1
        t0 = cute.arch.shuffle_sync_bfly(t0, mask)
        t1 = cute.arch.shuffle_sync_bfly(t1, mask)
        if high:
            a0, a1 = t0, t1
        else:
            b0, b1 = t0, t1
    return a0, a1, b0, b1


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
def load_matrix(
    data,
    tma,
    raw,
    shared,
    si,
    sm,
    fast,
    batch,
    tid,
    barrier,
    phase,
    atom_tma: cute.CopyAtom,
    width: cutlass.Constexpr,
    offset: cutlass.Constexpr,
    use_tma: cutlass.Constexpr,
):
    """Stage one selected KV matrix with TMA or predicated vector gathers."""
    num_fast = (fast[0] & 1) + (fast[1] & 1)
    if cutlass.const_expr(use_tma):
        if tid < 32 and num_fast > 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barrier.iterator, num_fast * 32 * width * 2)
            for piece in cutlass.range_constexpr(2):
                if (fast[piece] & 1) != 0:
                    token = si[piece * 32]
                    for col in cutlass.range_constexpr(width // 64):
                        src = cute.local_tile(
                            tma[None, None, batch], (32, 64), (token // 32, offset // 64 + col)
                        )
                        dst = cute.local_tile(shared, (32, 64), (piece, col))
                        copy, _, _ = copy_utils.tma_get_copy_fn(
                            atom_tma, 0, cute.make_layout(1), src, dst, single_stage=True
                        )
                        copy(tma_bar_ptr=barrier.iterator)
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), data.element_type, num_bits_per_copy=128)
    zero_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), data.element_type, num_bits_per_copy=128
    )
    zeros = cute.make_rmem_tensor((8,), data.element_type)
    zeros.fill(0)
    for piece in cutlass.range_constexpr(2):
        if (fast[piece] & 1) == 0:
            for item in cutlass.range_constexpr(width // 32):
                index = (tid + item * 128) * 8
                row, d = piece * 32 + index // width, index % width
                dst = vector_target(raw, row, d)
                if sm[row] != 0:
                    off = cute.crd2idx((batch, si[row], offset + d), data.layout)
                    src = cute.make_tensor((data.iterator + off).align(16), cute.make_layout((8,)))
                    cute.copy(atom, src, dst)
                else:
                    cute.copy(zero_atom, zeros, dst)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    if cutlass.const_expr(use_tma):
        if num_fast > 0:
            cute.arch.mbarrier_wait(barrier.iterator, phase)
            phase ^= 1
    cute.arch.sync_threads()
    cute.arch.fence_view_async_shared()
    return phase


# Attention kernel.


class PackedBackward:
    """Hopper backward over packed queries with FP32 atomic KV gradients."""

    def __init__(self, scale, heads=12, group=5, use_tma=True):
        self.scale, self.scale_log2 = scale, scale * 1.4426950408889634
        self.heads, self.group, self.use_tma = heads, group, use_tma

    @cute.jit
    def backward(
        self, q, k, v, do, ids, bases, bits, counts, lse, delta, dq, dk, dv, stream: cuda.CUstream
    ):
        """Build layouts and launch the packed sparse backward kernel."""
        qk = sm90_utils.make_tiled_mma(q.element_type, "K", "K", 64)
        dqm = sm90_utils.make_tiled_mma(q.element_type, "K", "MN", 256)
        dkm = sm90_utils.make_tiled_mma(q.element_type, "MN", "MN", 64)
        layout = sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (32, 64))
        ak, tk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(k, [1, 2, 0]), layout, (32, 64)
        )
        av, tv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(v, [1, 2, 0]), layout, (32, 64)
        )
        self.kernel(
            q,
            k,
            v,
            do,
            ids,
            bases,
            bits,
            counts,
            lse,
            delta,
            dq,
            dk,
            dv,
            tk,
            tv,
            qk,
            dqm,
            dkm,
            ak,
            av,
        ).launch(
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
        do,
        ids,
        bases,
        bits,
        counts,
        lse,
        delta,
        dq,
        dk,
        dv,
        tk,
        tv,
        mma_qk: cute.TiledMma,
        mma_dq: cute.TiledMma,
        mma_dkv: cute.TiledMma,
        atom_k: cute.CopyAtom,
        atom_v: cute.CopyAtom,
    ):
        """Recompute probabilities and accumulate dQ, dK and dV."""
        tid, _, _ = cute.arch.thread_idx()
        gi, batch, _ = cute.arch.block_idx()
        alloc = utils.SmemAllocator()
        sqr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        sdr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        skr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        svr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 128)),
            byte_alignment=1024,
        )
        spr = cute.make_tensor(
            svr.iterator,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 64)),
        )
        ssr = cute.make_tensor(
            svr.iterator + 4096,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 64)),
        )
        sq, sd, sk, sv, sp, ss = (
            swizzle_view(sqr),
            swizzle_view(sdr),
            swizzle_view(skr),
            swizzle_view(svr),
            swizzle_view(spr),
            swizzle_view(ssr),
        )
        si = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        sm = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        fast = alloc.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=8)
        barrier = alloc.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=8)
        if tid < 32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier.iterator, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        phase = Int32(0)
        cs = mma_qk.get_slice(tid).partition_C(cute.make_identity_tensor((64, 64)))
        cq = mma_dq.get_slice(tid).partition_C(cute.make_identity_tensor((64, 256)))
        cg = mma_dkv.get_slice(tid).partition_C(cute.make_identity_tensor((64, 64)))
        ap, fq, fk = sm90_utils.partition_fragment_ABC(mma_qk.get_slice(0), (64, 64, 256), sq, sk)
        adp = cute.make_rmem_tensor(cs.shape, Float32)
        aq, fds, fkt = sm90_utils.partition_fragment_ABC(
            mma_dq.get_slice(0), (64, 256, 64), ss, layout_utils.transpose_view(sk)
        )
        aq.fill(0.0)
        slse = cute.make_rmem_tensor(cs.shape, Float32)
        sdelta = cute.make_rmem_tensor(cs.shape, Float32)
        for i in cutlass.range_constexpr(cute.size(cs)):
            row, _ = cs[i]
            query = gi * self.group + row // self.heads
            lv, dval = Float32(0.0), Float32(0.0)
            if row < self.heads * self.group and query < q.shape[1]:
                lv = lse[batch, query, row % self.heads]
                dval = delta[batch, query, row % self.heads]
            slse[i], sdelta[i] = lv, dval
        load_packed_query(q, sqr, batch, gi, tid, self.heads, self.group)
        load_packed_query(do, sdr, batch, gi, tid, self.heads, self.group)
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
            phase = load_matrix(
                k,
                tk,
                skr,
                sk,
                si,
                sm,
                fast,
                batch,
                tid,
                barrier,
                phase,
                atom_k,
                256,
                0,
                self.use_tma,
            )
            sm90_utils.gemm(mma_qk, ap, fq, fk, zero_init=True, wg_wait=1)
            for half in cutlass.range_constexpr(2):
                phase = load_matrix(
                    v,
                    tv,
                    svr,
                    sv,
                    si,
                    sm,
                    fast,
                    batch,
                    tid,
                    barrier,
                    phase,
                    atom_v,
                    128,
                    half * 128,
                    self.use_tma,
                )
                do_half = cute.local_tile(sd, (64, 128), (0, half))
                fo = mma_qk.make_fragment_A(mma_qk.get_slice(0).partition_A(do_half))
                fv = mma_qk.make_fragment_B(mma_qk.get_slice(0).partition_B(sv))
                sm90_utils.gemm(mma_qk, adp, fo, fv, zero_init=(half == 0), wg_wait=0)
            for i in cutlass.range_constexpr(cute.size(cs)):
                row, col = cs[i]
                query = gi * self.group + row // self.heads
                member = Int32(1) << (row // self.heads)
                prob, ds = Float32(0.0), Float32(0.0)
                if row < self.group * self.heads and query < q.shape[1] and (sm[col] & member) != 0:
                    prob = cute.math.exp2(ap[i] * self.scale_log2 - slse[i])
                    ds = prob * (adp[i] - sdelta[i]) * self.scale
                sp[row, col] = prob.to(q.element_type)
                ss[row, col] = ds.to(q.element_type)
            cute.arch.sync_threads()
            cute.arch.fence_view_async_shared()
            sm90_utils.gemm(mma_dq, aq, fds, fkt, zero_init=False, wg_wait=0)
            for chunk in cutlass.range_constexpr(4):
                qt = cute.local_tile(layout_utils.transpose_view(sq), (64, 64), (chunk, 0))
                dot = cute.local_tile(layout_utils.transpose_view(sd), (64, 64), (chunk, 0))
                ak, ads, aqs = sm90_utils.partition_fragment_ABC(
                    mma_dkv.get_slice(0), (64, 64, 64), layout_utils.transpose_view(ss), qt
                )
                av, aps, ados = sm90_utils.partition_fragment_ABC(
                    mma_dkv.get_slice(0), (64, 64, 64), layout_utils.transpose_view(sp), dot
                )
                sm90_utils.gemm(mma_dkv, ak, ads, aqs, zero_init=True, wg_wait=1)
                sm90_utils.gemm(mma_dkv, av, aps, ados, zero_init=True, wg_wait=0)
                for rowpart in cutlass.range_constexpr(2):
                    for band in cutlass.range_constexpr(4):
                        ia = rowpart * 2 + band * 8
                        ib = ia + 4
                        row, d = cg[ia]
                        d += 2 * (tid % 4)
                        k0, k1, k2, k3 = contiguous_four(
                            ak[ia], ak[ia + 1], ak[ib], ak[ib + 1], tid
                        )
                        v0, v1, v2, v3 = contiguous_four(
                            av[ia], av[ia + 1], av[ib], av[ib + 1], tid
                        )
                        if sm[row] != 0:
                            off = cute.crd2idx((batch, si[row], chunk * 64 + d), dk.layout)
                            atomic_add4(dk.iterator + off, k0, k1, k2, k3)
                            atomic_add4(dv.iterator + off, v0, v1, v2, v3)
            cute.arch.sync_threads()
        cute.arch.cp_async_wait_group(0)
        for i in cutlass.range_constexpr(cute.size(cq)):
            row, d = cq[i]
            query = gi * self.group + row // self.heads
            if row < self.heads * self.group and query < q.shape[1]:
                dq[batch, query, row % self.heads, d] = aq[i].to(dq.element_type)
        if dq.shape[2] > self.heads:
            count = self.group * (dq.shape[2] - self.heads) * 256
            for i in cutlass.range(cute.ceil_div(count, 128)):
                j = tid + i * 128
                if j < count:
                    d = j % 256
                    hp = (j // 256) % (dq.shape[2] - self.heads)
                    query = gi * self.group + j // (256 * (dq.shape[2] - self.heads))
                    if query < dq.shape[1]:
                        dq[batch, query, self.heads + hp, d] = dq.element_type(0.0)
