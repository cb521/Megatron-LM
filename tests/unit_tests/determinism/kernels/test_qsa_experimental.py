# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay coverage for the standalone QSA kernels; backward atomics are not exact."""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


@pytest.fixture
def qsa():
    """Keep the experimental dependencies optional for ordinary Megatron tests."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
        pytest.skip("requires H100 or B200")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.cute_dsl_utils")
    pytest.importorskip("quack")
    from experimental.qsa_attention import qsa_kernels as api
    from experimental.qsa_attention.qsa_kernels import validation

    return api, validation


@pytest.mark.parametrize("heads", [3, 12])
@pytest.mark.parametrize("case", ["blocks", "arbitrary", "duplicates"])
@pytest.mark.parametrize("use_tma", [True, False])
def test_qsa_numerics(qsa, heads, case, use_tma):
    """Cover independent FP64 numerics, exact forward replay and additive gradients."""
    _, validation = qsa
    validation.validate_case(heads, case, use_tma)


def test_qsa_forward_replays_bit_exactly(qsa):
    """Replay packed routes and attention under concurrent GPU work."""
    api, validation = qsa
    q, k, v, _, ids, valid = validation.make_inputs(12, "duplicates", queries=257)
    plan = api.prepare_forward((q, k, v, ids, valid), heads=12)
    assert_replays_bit_exact(plan.run, (), replays=3, contention=True, what="QSA forward")


@pytest.mark.xfail(strict=False, reason="QSA backward uses unordered FP32 atomic dK/dV updates")
def test_qsa_backward_replay_status(qsa):
    """Record the known absence of a bit-exact backward contract."""
    api, validation = qsa
    q, k, v, do, ids, valid = validation.make_inputs(12, "duplicates", queries=257)
    out, lse = api.prepare_forward((q, k, v, ids, valid), heads=12).run()
    delta = (out.float() * do.float()).sum(-1).contiguous()
    plan = api.prepare_backward((q, k, v, do, ids, valid, lse, delta), heads=12)

    def run():
        plan.output[1].zero_()
        plan.output[2].zero_()
        return plan.run()

    assert_replays_bit_exact(run, (), replays=3, contention=True, what="QSA backward")
