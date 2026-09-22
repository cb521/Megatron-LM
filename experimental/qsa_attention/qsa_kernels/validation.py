"""Small independent FP64 checks for the installed package; run with ``python -m``."""

import logging
import math
import random

import torch

from .api import prepare_backward, prepare_forward


def make_inputs(heads: int, case: str, queries: int = 7) -> tuple[torch.Tensor, ...]:
    """Exercise empty rows, invalid IDs, holes, tails and shared or repeated routes."""
    generator = torch.Generator().manual_seed(9180 + heads)
    rng = random.Random(1921)
    q = torch.randn(1, queries, 16, 256, generator=generator, dtype=torch.bfloat16).cuda()
    k = torch.randn(1, 131, 256, generator=generator, dtype=torch.bfloat16).cuda()
    v = torch.randn(1, 131, 256, generator=generator, dtype=torch.bfloat16).cuda()
    do = torch.randn(1, queries, 16, 256, generator=generator, dtype=torch.bfloat16).cuda()
    q[:, :, heads:] = 0
    do[:, :, heads:] = 0
    ids = torch.full((1, queries, 129), -(2**31), dtype=torch.int32)
    valid = torch.zeros_like(ids)
    for qi in range(1, queries):
        route = [4 * block + i for block in rng.sample(range(32), 24) for i in range(4)]
        if case == "arbitrary":
            rng.shuffle(route)
        elif case == "duplicates":
            route[48:64] = route[:16]
        elif case != "blocks":
            raise ValueError(f"unknown case: {case}")
        route += [128, 129, 130]
        ids[0, qi, : len(route)] = torch.tensor(route, dtype=torch.int32)
        valid[0, qi, : len(route)] = 1
        if qi % 2:
            valid[0, qi, 10:13] = 0
    return q, k, v, do, ids.cuda(), valid.cuda()


def reference(q, k, v, ids, valid, do=None, lse=None, delta=None):
    """CPU FP64 attention; backward rounds P/dS to BF16 like the kernel contract."""
    q, k, v = (x.detach().cpu().double() for x in (q, k, v))
    ids, valid = ids.cpu().long(), valid.cpu().bool()
    out = torch.zeros_like(q)
    logsum = torch.full(q.shape[:-1], -math.inf, dtype=torch.float64)
    if do is not None:
        do, lse, delta = (x.detach().cpu().double() for x in (do, lse, delta))
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    scale = q.shape[-1] ** -0.5
    for b in range(q.shape[0]):
        for t in range(q.shape[1]):
            selected = ids[b, t, valid[b, t]]
            if selected.numel() == 0:
                continue
            keys, values = k[b, selected], v[b, selected]
            scores = (q[b, t] @ keys.T) * scale
            normalizer = torch.logsumexp(scores, dim=-1)
            probability = torch.exp(scores - normalizer[:, None])
            out[b, t] = probability @ values
            logsum[b, t] = normalizer / math.log(2)
            if do is not None:
                p = torch.exp2(scores / math.log(2) - lse[b, t, :, None])
                dp = do[b, t] @ values.T
                ds = (p * (dp - delta[b, t, :, None]) * scale).bfloat16().double()
                dq[b, t] = ds @ keys
                dk[b].index_add_(0, selected, ds.T @ q[b, t])
                dv[b].index_add_(0, selected, p.bfloat16().double().T @ do[b, t])
    if do is not None:
        return dq.bfloat16(), dk.float(), dv.float()
    return out, logsum


def assert_close(actual, expected, *, is_lse: bool = False) -> None:
    """Enforce fixed normalized error limits and identical infinity masks."""
    actual, expected = (x.detach().cpu().double() for x in (actual, expected))
    assert not torch.isnan(actual).any(), "NaN in kernel result"
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    if not finite.any():
        return
    error = actual[finite] - expected[finite]
    rms = max(expected[finite].square().mean().sqrt().item(), 1e-12)
    relative_l2 = error.square().mean().sqrt().item() / rms
    max_rms = error.abs().max().item() / rms
    limits = (1e-5, 1e-4) if is_lse else (0.008, 0.12)
    assert (
        relative_l2 <= limits[0] and max_rms <= limits[1]
    ), f"relative L2={relative_l2:.6g}, max/RMS={max_rms:.6g}, limits={limits}"


def validate_case(heads: int, case: str, use_tma: bool) -> None:
    """Check forward, backward, accumulation and live route updates on this GPU."""
    q, k, v, do, ids, valid = make_inputs(heads, case)
    expected_out, expected_lse = reference(q[:, :, :heads], k, v, ids, valid)
    fwd = prepare_forward((q, k, v, ids, valid), heads=heads, use_tma=use_tma)
    out, lse = fwd.run()
    assert_close(out[:, :, :heads], expected_out)
    assert_close(lse[:, :, :heads], expected_lse, is_lse=True)
    snapshots = tuple(x.clone() for x in fwd.output)
    for _ in range(3):
        for actual, expected in zip(fwd.run(), snapshots):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    delta = (out.float() * do.float()).sum(-1).contiguous()
    expected_grads = reference(
        q[:, :, :heads], k, v, ids, valid, do[:, :, :heads], lse[:, :, :heads], delta[:, :, :heads]
    )
    dk = torch.full_like(k, 0.125, dtype=torch.float32)
    dv = torch.full_like(v, -0.125, dtype=torch.float32)
    bwd = prepare_backward(
        (q, k, v, do, ids, valid, lse, delta), heads=heads, use_tma=use_tma, dk=dk, dv=dv
    )
    bwd.output[0].fill_(math.nan)
    dq, _, _ = bwd.run()
    assert torch.isfinite(dq).all(), "backward did not overwrite every dQ element"
    for actual, expected in zip((dq[:, :, :heads], dk - 0.125, dv + 0.125), expected_grads):
        assert_close(actual, expected)
    bwd.run()
    assert_close(dk, 0.125 + 2 * expected_grads[1])
    assert_close(dv, -0.125 + 2 * expected_grads[2])
    valid[:, 3] = 0
    changed, _ = reference(q[:, :, :heads], k, v, ids, valid)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fwd.run()
    torch.cuda.current_stream().wait_stream(stream)
    assert_close(fwd.output[0][:, :, :heads], changed)


def main() -> None:
    """Run both payload paths for both measured head counts."""
    torch.set_num_threads(8)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("%s", torch.cuda.get_device_name())
    for heads in (3, 12):
        for case in ("blocks", "arbitrary", "duplicates"):
            for use_tma in (True, False):
                validate_case(heads, case, use_tma)
                logging.info("PASS heads=%s routes=%s tma=%s", heads, case, use_tma)


if __name__ == "__main__":
    main()
