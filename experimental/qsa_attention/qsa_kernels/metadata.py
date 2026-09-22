"""Build exact token membership, with an occurrence-preserving fallback."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_loaded = False


def metadata(ids, valid, skv, group):
    """Allocate KV-union metadata and return its current-stream update callable."""
    global _loaded
    if not _loaded:
        load(
            name="qsa_routes_clean_v3",
            sources=[str(Path(__file__).with_name("metadata.cu"))],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            is_python_module=False,
            verbose=True,
        )
        _loaded = True
    cap = max(128, 1 << (group * ((ids.shape[2] + 31) // 32) - 1).bit_length())
    shape = (ids.shape[0], (ids.shape[1] + group - 1) // group)
    bases = torch.empty((*shape, cap), dtype=torch.int32, device=ids.device)
    bits = torch.empty((*shape, cap, group), dtype=torch.int32, device=ids.device)
    counts = torch.empty(shape, dtype=torch.int32, device=ids.device)

    def run():
        torch.ops.qsa_routes_clean_v3.build(ids, valid, bases, bits, counts, group, skv)

    run()
    return (bases, bits, counts), run
