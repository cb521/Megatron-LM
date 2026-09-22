# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


"""Blackwell M128 PV with CTA-uniform fully masked tile skipping."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cutlass_dsl import dsl_user_op
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
def load_query(q, sq, batch, qi, hbase, tid):
    """Stage one query and 16 heads using asynchronous vector copies."""
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), q.element_type, num_bits_per_copy=128)
    for item in cutlass.range_constexpr(4):
        index = (tid + item * 128) * 8
        h, d = index // 256, index % 256
        off = cute.crd2idx((batch, qi, hbase + h, d), q.layout)
        src = cute.make_tensor((q.iterator + off).align(16), cute.make_layout((8,)))
        cute.copy(atom, src, vector_target(sq, h, d))


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
def read_metadata(ids, valid, batch, qi, tile, tid):
    """Prefetch one tile of selected IDs and validity flags into registers."""
    token, mask = Int32(0), Int32(0)
    if tid < 64:
        slot = tile * 64 + tid
        if slot < ids.shape[2]:
            token = ids[batch, qi, slot]
            mask = valid[batch, qi, slot]
    return token, mask


@cute.jit
def publish_metadata(token, mask, si, sm, fast, tid, use_tma: cutlass.Constexpr):
    """Publish routes and identify fully contiguous chunks eligible for TMA."""
    if tid < 64:
        si[tid], sm[tid] = token, mask
        first = cute.arch.shuffle_sync(token, 0)
        good = (mask != 0) & (token == first + tid % 32) & (first % 32 == 0)
        unanimous = cute.arch.vote_all_sync(good)
        any_valid = cute.arch.vote_ballot_sync(mask != 0) != 0
        if tid % 32 == 0:
            tma_flag = Int32(unanimous) if cutlass.const_expr(use_tma) else Int32(0)
            fast[tid // 32] = tma_flag | (Int32(any_valid) << 1)
    cute.arch.sync_threads()
    return ((fast[0] | fast[1]) & 2) != 0


@dsl_user_op
def tmem_before_sync(*, loc=None, ip=None):
    """Order tensor-memory operations before a thread synchronization."""
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::before_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_after_sync(*, loc=None, ip=None):
    """Order tensor-memory operations after a thread synchronization."""
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


# Attention kernel.


class TokenForward:
    """Blackwell attention using swapAB with one query per CTA/head group."""

    def __init__(self, scale, use_tma=True):
        self.scale_log2 = scale * 1.4426950408889634
        self.use_tma = use_tma

    @cute.jit
    def forward(self, q, k, v, ids, valid, out, lse, stream: cuda.CUstream):
        """Build tensor-memory layouts and launch the swapAB forward kernel."""
        qk = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                Float32,
                (64, 16, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        vp = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                Float32,
                (128, 16, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.MN,
                OperandMajorMode.K,
            )
        )
        layout = sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (32, 64))
        ak, tk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(k, [1, 2, 0]), layout, (32, 64)
        )
        av, tv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(v, [1, 2, 0]), layout, (32, 64)
        )
        self.kernel(q, k, v, tk, tv, ids, valid, out, lse, qk, vp, ak, av).launch(
            grid=(q.shape[1], q.shape[0], q.shape[2] // 16),
            block=(128, 1, 1),
            stream=stream,
            min_blocks_per_mp=3,
        )

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        tk,
        tv,
        ids,
        valid,
        out,
        lse,
        mma_qk: cute.TiledMma,
        mma_vp: cute.TiledMma,
        atom_k: cute.CopyAtom,
        atom_v: cute.CopyAtom,
    ):
        """Compute sparse attention with transposed QK and PV GEMMs."""
        tid, _, _ = cute.arch.thread_idx()
        qi, batch, hb = cute.arch.block_idx()
        hbase = hb * 16
        alloc = utils.SmemAllocator()
        barrier = alloc.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=8)
        mma_barrier = alloc.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=8)
        tmem_address = alloc.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
        if tid < 32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier.iterator, 1)
                cute.arch.mbarrier_init(mma_barrier.iterator, 1)
        cute.arch.mbarrier_init_fence()
        tmem_before_sync()
        cute.arch.sync_threads()
        tmem_after_sync()
        phase = Int32(0)
        mma_phase = Int32(0)

        sq_raw = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (16, 256)),
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
        # K is dead after the QK completion wait and before the next tile load.
        # Scores [0,4576) and P [8192,10240) use disjoint bytes of that storage.
        # Keep the existing QK/PV waits, CTA barriers and async fences unchanged.
        sp_raw = cute.make_tensor(
            sk_raw.iterator + 4096,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (16, 64)),
        )
        sq, sk, sv, sp = (
            swizzle_view(sq_raw),
            swizzle_view(sk_raw),
            swizzle_view(sv_raw),
            swizzle_view(sp_raw),
        )
        scores = cute.make_tensor(
            cute.recast_ptr(sk_raw.iterator, dtype=Float32),
            cute.make_layout((16, 64), stride=(72, 1)),
        )
        row_scale = alloc.allocate_tensor(Float32, cute.make_layout((16,)), byte_alignment=128)
        row_sum = alloc.allocate_tensor(Float32, cute.make_layout((16,)), byte_alignment=128)
        si = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        sm = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        fast = alloc.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=8)

        frag_k = mma_qk.make_fragment_A(mma_qk.get_slice(0).partition_A(sk))
        frag_q = mma_qk.make_fragment_B(mma_qk.get_slice(0).partition_B(sq))
        frag_v = mma_vp.make_fragment_A(
            mma_vp.get_slice(0).partition_A(layout_utils.transpose_view(sv))
        )
        frag_p = mma_vp.make_fragment_B(mma_vp.get_slice(0).partition_B(sp))
        fake_s = mma_qk.make_fragment_C(mma_qk.partition_shape_C((64, 16)))
        fake_o = mma_vp.make_fragment_C(mma_vp.partition_shape_C((256, 16)))
        tmem_cols = utils.get_num_tmem_alloc_cols(fake_o)
        if tid < 32:
            cute.arch.alloc_tmem(tmem_cols, tmem_address.iterator, is_two_cta=False)
        tmem_before_sync()
        cute.arch.sync_threads()
        tmem_after_sync()
        tptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_address.iterator
        )
        t_s = cute.make_tensor(tptr, fake_s.layout)
        t_o = cute.make_tensor(tptr, fake_o.layout)
        if tid < 32:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        # QK and PV reuse TMEM after every participating warp has finished reading.
        t_s_matrix = cute.make_tensor(t_s.iterator, t_s.layout[0])
        load_atom = cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32)
        load_copy = tcgen05.make_tmem_copy(load_atom, t_s_matrix)
        load_slice = load_copy.get_slice(tid)
        t_s_src = load_slice.partition_S(t_s_matrix)
        coords = load_slice.partition_D(cute.make_identity_tensor((64, 16)))
        acc_s = cute.make_rmem_tensor(coords.shape, Float32)
        # The QK score remains M64; PV uses a separate M128 TMEM copy layout.
        t_o_matrix0 = cute.make_tensor(t_o.iterator, t_o.layout[0])
        o_load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), Float32)
        o_load_copy = tcgen05.make_tmem_copy(o_load_atom, t_o_matrix0)
        o_load_slice = o_load_copy.get_slice(tid)
        o_coords = o_load_slice.partition_D(cute.make_identity_tensor((128, 16)))
        o_reg_layout = cute.make_layout(o_coords.shape)
        acc_o = cute.make_rmem_tensor((cute.size(o_coords), 2), Float32)
        acc_o.fill(0.0)
        head, lane = tid // 8, tid % 8
        running_max, denominator = Float32(-float("inf")), Float32(0.0)
        probability = cute.make_rmem_tensor((8,), Float32)
        load_query(q, sq_raw, batch, qi, hbase, tid)
        cute.arch.cp_async_commit_group()

        next_token, next_mask = read_metadata(ids, valid, batch, qi, Int32(0), tid)
        for tile in cutlass.range(cute.ceil_div(ids.shape[2], 64), unroll=1):
            has_work = publish_metadata(next_token, next_mask, si, sm, fast, tid, self.use_tma)
            # Read into registers only. Current shared metadata stays live through PV.
            next_token, next_mask = read_metadata(ids, valid, batch, qi, tile + 1, tid)
            if has_work:
                # Recreate compile-time handles in the loop: avoids opaque loop-carried
                # values in the pinned DSL. No arithmetic or synchronization is changed.
                qk_step = cute.make_tiled_mma(
                    tcgen05.MmaF16BF16Op(
                        q.element_type,
                        Float32,
                        (64, 16, 16),
                        tcgen05.CtaGroup.ONE,
                        tcgen05.OperandSource.SMEM,
                        OperandMajorMode.K,
                        OperandMajorMode.K,
                    )
                )
                vp_step = cute.make_tiled_mma(
                    tcgen05.MmaF16BF16Op(
                        q.element_type,
                        Float32,
                        (128, 16, 16),
                        tcgen05.CtaGroup.ONE,
                        tcgen05.OperandSource.SMEM,
                        OperandMajorMode.MN,
                        OperandMajorMode.K,
                    )
                )
                load_step = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32),
                    t_s_matrix,
                )
                o_load_step = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), Float32),
                    t_o_matrix0,
                )
                o_store_step = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), Float32),
                    t_o_matrix0,
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
                if tid < 32:
                    qk_step.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(cute.size(frag_k, mode=[2])):
                        cute.gemm(qk_step, t_s, frag_k[None, None, kk], frag_q[None, None, kk], t_s)
                        qk_step.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mma_barrier.iterator)
                cute.arch.mbarrier_wait(mma_barrier.iterator, mma_phase)
                tmem_after_sync()
                mma_phase ^= 1
                cute.copy(load_step, t_s_src, acc_s)
                cute.arch.fence_view_async_tmem_op("load")
                for i in cutlass.range_constexpr(cute.size(acc_s)):
                    row, h = coords[i]
                    scores[h, row] = acc_s[i]
                tmem_before_sync()
                cute.arch.sync_threads()
                tmem_after_sync()

                tile_max = Float32(-float("inf"))
                for item in cutlass.range_constexpr(8):
                    row = lane + item * 8
                    value = Float32(-float("inf"))
                    if sm[row] != 0:
                        value = scores[head, row] * self.scale_log2
                    probability[item] = value
                    tile_max = cute.arch.fmax(tile_max, value)
                for delta in cutlass.range_constexpr(3):
                    tile_max = cute.arch.fmax(
                        tile_max, cute.arch.shuffle_sync_bfly(tile_max, 1 << delta)
                    )
                next_max = cute.arch.fmax(running_max, tile_max)
                alpha = Float32(1.0)
                if next_max != -float("inf"):
                    alpha = cute.math.exp2(running_max - next_max)
                total = Float32(0.0)
                for item in cutlass.range_constexpr(8):
                    prob = Float32(0.0)
                    if probability[item] != -float("inf"):
                        prob = cute.math.exp2(probability[item] - next_max)
                    sp[head, lane + item * 8] = prob.to(q.element_type)
                    total += prob
                for delta in cutlass.range_constexpr(3):
                    total += cute.arch.shuffle_sync_bfly(total, 1 << delta)
                denominator = denominator * alpha + total
                running_max = next_max
                if lane == 0:
                    row_scale[head] = alpha
                tmem_before_sync()
                cute.arch.sync_threads()
                tmem_after_sync()

                # Preserve FP32 online accumulation: scale old O, seed TMEM with it,
                # then let tcgen05 accumulate P@V into that same accumulator.
                for chunk in cutlass.range_constexpr(2):
                    for i in cutlass.range_constexpr(cute.size(o_coords)):
                        _, h = o_coords[i]
                        acc_o[i, chunk] *= row_scale[h]
                    r_o = cute.make_tensor(acc_o[None, chunk].iterator, o_reg_layout)
                    t_o_matrix = cute.make_tensor(t_o[None, chunk, 0].iterator, t_o.layout[0])
                    cute.copy(
                        o_store_step, r_o, o_store_step.get_slice(tid).partition_D(t_o_matrix)
                    )
                cute.arch.fence_view_async_tmem_op("store")
                tmem_before_sync()
                cute.arch.sync_threads()
                tmem_after_sync()
                cute.arch.fence_view_async_shared()
                if tid < 32:
                    vp_step.set(tcgen05.Field.ACCUMULATE, True)
                    for kk in cutlass.range_constexpr(cute.size(frag_v, mode=[2])):
                        cute.gemm(vp_step, t_o, frag_v[None, None, kk], frag_p[None, None, kk], t_o)
                    with cute.arch.elect_one():
                        tcgen05.commit(mma_barrier.iterator)
                cute.arch.mbarrier_wait(mma_barrier.iterator, mma_phase)
                tmem_after_sync()
                mma_phase ^= 1
                for chunk in cutlass.range_constexpr(2):
                    r_o = cute.make_tensor(acc_o[None, chunk].iterator, o_reg_layout)
                    t_o_matrix = cute.make_tensor(t_o[None, chunk, 0].iterator, t_o.layout[0])
                    cute.copy(o_load_step, o_load_step.get_slice(tid).partition_S(t_o_matrix), r_o)
                cute.arch.fence_view_async_tmem_op("load")
            # All warps consume this tile decision before metadata is reused.
            tmem_before_sync()
            cute.arch.sync_threads()
            tmem_after_sync()
        cute.arch.cp_async_wait_group(0)
        if lane == 0:
            row_sum[head] = denominator
            logsum = Float32(-float("inf"))
            if denominator > 0.0:
                logsum = running_max + cute.math.log2(denominator)
            lse[batch, qi, hbase + head] = logsum
        tmem_before_sync()
        cute.arch.sync_threads()
        tmem_after_sync()
        for chunk in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(cute.size(o_coords)):
                d, h = o_coords[i]
                result = Float32(0.0)
                if row_sum[h] > 0.0:
                    result = acc_o[i, chunk] / row_sum[h]
                out[batch, qi, hbase + h, chunk * 128 + d] = result.to(out.element_type)
        tmem_before_sync()
        cute.arch.sync_threads()
        tmem_after_sync()
        if tid < 32:
            cute.arch.dealloc_tmem(tptr, tmem_cols, is_two_cta=False)
