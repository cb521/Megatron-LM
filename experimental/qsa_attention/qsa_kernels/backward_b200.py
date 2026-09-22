# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


"""Single-core Blackwell port of query-owned five-GEMM backward.

Original prepared ABI, BF16 P/dS, FP32 accumulators and additive FP32 dK/dV.
QK and dP are AB-swapped; dQ uses M128 slices, dK/dV M64xN64.
"""

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


class TokenBackward:
    """Blackwell backward with swapAB for QK, dP and dQ GEMMs."""

    def __init__(self, scale, use_tma=True):
        self.scale, self.scale_log2 = scale, scale * 1.4426950408889634
        self.use_tma = use_tma

    @cute.jit
    def backward(self, q, k, v, do, ids, valid, lse, delta, dq, dk, dv, stream: cuda.CUstream):
        """Build tensor-memory layouts and launch sparse attention backward."""
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
        dqm = cute.make_tiled_mma(
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
        dkm = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                Float32,
                (64, 64, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.MN,
                OperandMajorMode.MN,
            )
        )
        layout = sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (32, 64))
        ak, tk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(k, [1, 2, 0]), layout, (32, 64)
        )
        av, tv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), layout_utils.select(v, [1, 2, 0]), layout, (32, 64)
        )
        self.kernel(
            q, k, v, do, ids, valid, lse, delta, dq, dk, dv, tk, tv, qk, dqm, dkm, ak, av
        ).launch(
            grid=(q.shape[1], q.shape[0], q.shape[2] // 16),
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
        valid,
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
        """Compute five backward GEMMs and atomically accumulate KV gradients."""
        tid, _, _ = cute.arch.thread_idx()
        qi, batch, hb = cute.arch.block_idx()
        hbase = hb * 16
        alloc = utils.SmemAllocator()
        barrier = alloc.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=8)
        mma_barrier = alloc.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=8)
        address = alloc.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=4)
        if tid < 32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(barrier.iterator, 1)
                cute.arch.mbarrier_init(mma_barrier.iterator, 1)
            cute.arch.alloc_tmem(128, address.iterator, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        tmem_before_sync()
        cute.arch.sync_threads()
        tmem_after_sync()
        tptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=address.iterator
        )
        if tid < 32:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        phase, mma_phase = Int32(0), Int32(0)
        sqr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (16, 256)),
            byte_alignment=1024,
        )
        sdr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (16, 256)),
            byte_alignment=1024,
        )
        skr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        svr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (64, 256)),
            byte_alignment=1024,
        )
        spr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (16, 64)),
            byte_alignment=1024,
        )
        ssr = alloc.allocate_tensor(
            q.element_type,
            sm90_utils.make_smem_layout(q.element_type, LayoutEnum.ROW_MAJOR, (16, 64)),
            byte_alignment=1024,
        )
        sq, sd, sk, sv, sp, ss = (
            swizzle_view(sqr),
            swizzle_view(sdr),
            swizzle_view(skr),
            swizzle_view(svr),
            swizzle_view(spr),
            swizzle_view(ssr),
        )
        # Four FP32 padding entries per row reduce bank conflicts for this
        # M64 TMEM fragment and preserve 16-byte alignment for vector atomics.
        gradk = cute.make_tensor(
            cute.recast_ptr(skr.iterator, dtype=Float32), cute.make_layout((64, 64), stride=(68, 1))
        )
        gradv = cute.make_tensor(
            cute.recast_ptr(svr.iterator, dtype=Float32), cute.make_layout((64, 64), stride=(68, 1))
        )
        si = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        sm = alloc.allocate_tensor(Int32, cute.make_layout((64,)), byte_alignment=128)
        fast = alloc.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=8)
        fs = mma_qk.make_fragment_C(mma_qk.partition_shape_C((64, 16)))
        fdq = mma_dq.make_fragment_C(mma_dq.partition_shape_C((128, 16)))
        fg = mma_dkv.make_fragment_C(mma_dkv.partition_shape_C((64, 64)))
        ts = cute.make_tensor(tptr, fs.layout)
        tdp = cute.make_tensor(tptr + 16, fs.layout)
        tdq = cute.make_tensor(tptr, fdq.layout)
        tgk = cute.make_tensor(tptr, fg.layout)
        tgv = cute.make_tensor(tptr + 64, fg.layout)
        ms, mdp = (
            cute.make_tensor(ts.iterator, ts.layout[0]),
            cute.make_tensor(tdp.iterator, tdp.layout[0]),
        )
        mdq = cute.make_tensor(tdq.iterator, tdq.layout[0])
        mgk, mgv = (
            cute.make_tensor(tgk.iterator, tgk.layout[0]),
            cute.make_tensor(tgv.iterator, tgv.layout[0]),
        )
        ls0 = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32), ms
        )
        ldq0 = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), Float32), mdq
        )
        lg0 = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32), mgk
        )
        cs = ls0.get_slice(tid).partition_D(cute.make_identity_tensor((64, 16)))
        cdq = ldq0.get_slice(tid).partition_D(cute.make_identity_tensor((128, 16)))
        cg = lg0.get_slice(tid).partition_D(cute.make_identity_tensor((64, 64)))
        acc_q = cute.make_rmem_tensor((cute.size(cdq), 2), Float32)
        acc_q.fill(0.0)
        qreg_layout = cute.make_layout(cdq.shape)
        saved_lse = cute.make_rmem_tensor(cs.shape, Float32)
        saved_delta = cute.make_rmem_tensor(cs.shape, Float32)
        for i in cutlass.range_constexpr(cute.size(cs)):
            _, h = cs[i]
            saved_lse[i] = lse[batch, qi, hbase + h]
            saved_delta[i] = delta[batch, qi, hbase + h]
        load_query(q, sqr, batch, qi, hbase, tid)
        load_query(do, sdr, batch, qi, hbase, tid)
        cute.arch.cp_async_commit_group()
        next_token, next_mask = read_metadata(ids, valid, batch, qi, Int32(0), tid)
        for tile in cutlass.range(cute.ceil_div(ids.shape[2], 64), unroll=1):
            has_work = publish_metadata(next_token, next_mask, si, sm, fast, tid, self.use_tma)
            next_token, next_mask = read_metadata(ids, valid, batch, qi, tile + 1, tid)
            if has_work:
                phase = load_payload(
                    k,
                    v,
                    tk,
                    tv,
                    skr,
                    svr,
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
                fk, fq = (
                    qk.make_fragment_A(qk.get_slice(0).partition_A(sk)),
                    qk.make_fragment_B(qk.get_slice(0).partition_B(sq)),
                )
                fv, fo = (
                    qk.make_fragment_A(qk.get_slice(0).partition_A(sv)),
                    qk.make_fragment_B(qk.get_slice(0).partition_B(sd)),
                )
                if tid < 32:
                    qk.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(cute.size(fk, mode=[2])):
                        cute.gemm(qk, ts, fk[None, None, kk], fq[None, None, kk], ts)
                        qk.set(tcgen05.Field.ACCUMULATE, True)
                    qk.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(cute.size(fv, mode=[2])):
                        cute.gemm(qk, tdp, fv[None, None, kk], fo[None, None, kk], tdp)
                        qk.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mma_barrier.iterator)
                cute.arch.mbarrier_wait(mma_barrier.iterator, mma_phase)
                tmem_after_sync()
                mma_phase ^= 1
                ls = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32), ms
                )
                lp = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32), mdp
                )
                ap, adp = (
                    cute.make_rmem_tensor(cs.shape, Float32),
                    cute.make_rmem_tensor(cs.shape, Float32),
                )
                cute.copy(ls, ls.get_slice(tid).partition_S(ms), ap)
                cute.copy(lp, lp.get_slice(tid).partition_S(mdp), adp)
                cute.arch.fence_view_async_tmem_op("load")
                for i in cutlass.range_constexpr(cute.size(cs)):
                    row, h = cs[i]
                    prob, ds = Float32(0.0), Float32(0.0)
                    if sm[row] != 0:
                        prob = cute.math.exp2(ap[i] * self.scale_log2 - saved_lse[i])
                        ds = prob * (adp[i] - saved_delta[i]) * self.scale
                    sp[h, row] = prob.to(q.element_type)
                    ss[h, row] = ds.to(q.element_type)
                tmem_before_sync()
                cute.arch.sync_threads()
                tmem_after_sync()
                cute.arch.fence_view_async_shared()
                for chunk in cutlass.range_constexpr(2):
                    dqm = cute.make_tiled_mma(
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
                    kt = cute.local_tile(layout_utils.transpose_view(sk), (128, 64), (chunk, 0))
                    fkt = dqm.make_fragment_A(dqm.get_slice(0).partition_A(kt))
                    fds = dqm.make_fragment_B(dqm.get_slice(0).partition_B(ss))
                    ldq = tcgen05.make_tmem_copy(
                        cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), Float32),
                        mdq,
                    )
                    sdq = tcgen05.make_tmem_copy(
                        cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), Float32),
                        mdq,
                    )
                    rq = cute.make_tensor(acc_q[None, chunk].iterator, qreg_layout)
                    cute.copy(sdq, rq, sdq.get_slice(tid).partition_D(mdq))
                    cute.arch.fence_view_async_tmem_op("store")
                    tmem_before_sync()
                    cute.arch.sync_threads()
                    tmem_after_sync()
                    if tid < 32:
                        dqm.set(tcgen05.Field.ACCUMULATE, True)
                        for kk in cutlass.range_constexpr(cute.size(fkt, mode=[2])):
                            cute.gemm(dqm, tdq, fkt[None, None, kk], fds[None, None, kk], tdq)
                        with cute.arch.elect_one():
                            tcgen05.commit(mma_barrier.iterator)
                    cute.arch.mbarrier_wait(mma_barrier.iterator, mma_phase)
                    tmem_after_sync()
                    mma_phase ^= 1
                    cute.copy(ldq, ldq.get_slice(tid).partition_S(mdq), rq)
                    cute.arch.fence_view_async_tmem_op("load")
                    tmem_before_sync()
                    cute.arch.sync_threads()
                    tmem_after_sync()
                # K/V are dead here. Reuse their shared storage for gradients.
                for chunk in cutlass.range_constexpr(4):
                    dkm = cute.make_tiled_mma(
                        tcgen05.MmaF16BF16Op(
                            q.element_type,
                            Float32,
                            (64, 64, 16),
                            tcgen05.CtaGroup.ONE,
                            tcgen05.OperandSource.SMEM,
                            OperandMajorMode.MN,
                            OperandMajorMode.MN,
                        )
                    )
                    qt = cute.local_tile(layout_utils.transpose_view(sq), (64, 16), (chunk, 0))
                    dot = cute.local_tile(layout_utils.transpose_view(sd), (64, 16), (chunk, 0))
                    fds = dkm.make_fragment_A(
                        dkm.get_slice(0).partition_A(layout_utils.transpose_view(ss))
                    )
                    fp = dkm.make_fragment_A(
                        dkm.get_slice(0).partition_A(layout_utils.transpose_view(sp))
                    )
                    fq = dkm.make_fragment_B(dkm.get_slice(0).partition_B(qt))
                    fdo = dkm.make_fragment_B(dkm.get_slice(0).partition_B(dot))
                    if tid < 32:
                        dkm.set(tcgen05.Field.ACCUMULATE, False)
                        cute.gemm(dkm, tgk, fds[None, None, 0], fq[None, None, 0], tgk)
                        cute.gemm(dkm, tgv, fp[None, None, 0], fdo[None, None, 0], tgv)
                        with cute.arch.elect_one():
                            tcgen05.commit(mma_barrier.iterator)
                    cute.arch.mbarrier_wait(mma_barrier.iterator, mma_phase)
                    tmem_after_sync()
                    mma_phase ^= 1
                    lk = tcgen05.make_tmem_copy(
                        cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32),
                        mgk,
                    )
                    lv = tcgen05.make_tmem_copy(
                        cute.make_copy_atom(tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), Float32),
                        mgv,
                    )
                    ak, av = (
                        cute.make_rmem_tensor(cg.shape, Float32),
                        cute.make_rmem_tensor(cg.shape, Float32),
                    )
                    cute.copy(lk, lk.get_slice(tid).partition_S(mgk), ak)
                    cute.copy(lv, lv.get_slice(tid).partition_S(mgv), av)
                    cute.arch.fence_view_async_tmem_op("load")
                    for i in cutlass.range_constexpr(cute.size(cg)):
                        row, d = cg[i]
                        gradk[row, d] = ak[i]
                        gradv[row, d] = av[i]
                    tmem_before_sync()
                    cute.arch.sync_threads()
                    tmem_after_sync()
                    for item in cutlass.range_constexpr(8):
                        index = (tid + item * 128) * 4
                        row, d = index // 64, index % 64
                        if sm[row] != 0:
                            off = cute.crd2idx((batch, si[row], chunk * 64 + d), dk.layout)
                            atomic_add4(
                                dk.iterator + off,
                                gradk[row, d],
                                gradk[row, d + 1],
                                gradk[row, d + 2],
                                gradk[row, d + 3],
                            )
                            atomic_add4(
                                dv.iterator + off,
                                gradv[row, d],
                                gradv[row, d + 1],
                                gradv[row, d + 2],
                                gradv[row, d + 3],
                            )
                    tmem_before_sync()
                    cute.arch.sync_threads()
                    tmem_after_sync()
            tmem_before_sync()
            cute.arch.sync_threads()
            tmem_after_sync()
        cute.arch.cp_async_wait_group(0)
        for chunk in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(cute.size(cdq)):
                d, head_out = cdq[i]
                dq[batch, qi, hbase + head_out, chunk * 128 + d] = acc_q[i, chunk].to(
                    dq.element_type
                )
        tmem_before_sync()
        cute.arch.sync_threads()
        tmem_after_sync()
        if tid < 32:
            cute.arch.dealloc_tmem(tptr, 128, is_two_cta=False)
