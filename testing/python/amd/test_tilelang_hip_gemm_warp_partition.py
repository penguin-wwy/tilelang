import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.rocm.intrinsics.wmma_macro_generator import WMMAIntrinEmitter
from tilelang.rocm.target import target_get_warp_size, target_is_cdna, target_is_rdna


POLICIES = [T.GemmWarpPolicy.Square, T.GemmWarpPolicy.FullRow, T.GemmWarpPolicy.FullCol]


def _warp_partition(m, n, num_warps, policy, arch):
    target = determine_target({"kind": "hip", "mcpu": arch}, return_object=True)
    node = tvm.ir.make_node("tl.GemmWarpPolicy", policy_type=int(policy), m_warp=0, n_warp=0)
    instruction = "rocm.mfma" if target_is_cdna(target) else "rocm.wmma"
    return node.compute_warp_partition(m, n, num_warps * target_get_warp_size(target), target, instruction)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx942"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    "m,n,num_warps,expected",
    [
        (64, 48, 4, (4, 1)),
        (48, 64, 4, (1, 4)),
        (64, 80, 4, (4, 1)),
        (128, 48, 4, (4, 1)),
        (64, 48, 6, (2, 3)),
        (64, 48, 2, (2, 1)),
        (64, 48, 3, (1, 3)),
    ],
)
def test_gemm_warp_partition_covers_complete_tiles(m, n, num_warps, expected, policy, arch):
    assert _warp_partition(m, n, num_warps, policy, arch) == expected


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx942"])
@pytest.mark.parametrize("policy,expected", [(POLICIES[0], (2, 2)), (POLICIES[1], (4, 1)), (POLICIES[2], (1, 4))])
def test_gemm_warp_partition_preserves_policy(policy, expected, arch):
    assert _warp_partition(64, 64, 4, policy, arch) == expected


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx942"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("m,n,num_warps", [(64, 48, 8), (48, 48, 4), (16, 16, 2)])
def test_gemm_warp_partition_rejects_incomplete_tiles(m, n, num_warps, policy, arch):
    with pytest.raises(tvm.error.InternalError, match="No valid warp partition.*adjust `threads` or the block tile shape"):
        _warp_partition(m, n, num_warps, policy, arch)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
@pytest.mark.parametrize("dimension", ["warp_row_tiles", "warp_col_tiles"])
@pytest.mark.parametrize("tiles", [0, 8, 24])
def test_wmma_emitter_rejects_partial_tiles(arch, dimension, tiles):
    target = tvm.target.Target({"kind": "hip", "mcpu": arch})
    with pytest.raises(ValueError, match=rf"{dimension} must be a positive multiple of 16"):
        WMMAIntrinEmitter(target=target, **{dimension: tiles})


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_wmma_emitter_accepts_complete_tiles(arch):
    target = tvm.target.Target({"kind": "hip", "mcpu": arch})
    emitter = WMMAIntrinEmitter(target=target, warp_row_tiles=32, warp_col_tiles=48)
    assert emitter.warp_rows == 2
    assert emitter.warp_cols == 3


def _gemm_kernel(m, n, threads, policy, transpose_b=False):
    k = 16
    b_shape = (n, k) if transpose_b else (k, n)

    @T.prim_func
    def main(A: T.Tensor((m, k), "float16"), B: T.Tensor(b_shape, "float16"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1, threads=threads):
            a = T.alloc_shared((m, k), "float16")
            b = T.alloc_shared(b_shape, "float16")
            c = T.alloc_fragment((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            T.gemm(a, b, c, transpose_B=transpose_b, policy=policy)
            T.copy(c, C)

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx942"])
def test_gemm_rejects_uncoverable_tile_before_codegen(arch):
    target = determine_target({"kind": "hip", "mcpu": arch}, return_object=True)
    program = _gemm_kernel(64, 48, 8 * target_get_warp_size(target), T.GemmWarpPolicy.Square)
    with tvm.transform.PassContext(), target, pytest.raises(tvm.error.InternalError, match="No valid warp partition"):
        tilelang.lower(program, target=target, enable_device_compile=False)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize(
    "m,n,threads,policy",
    [
        (64, 48, 128, T.GemmWarpPolicy.Square),
        (64, 48, 128, T.GemmWarpPolicy.FullRow),
        (64, 48, 128, T.GemmWarpPolicy.FullCol),
        (48, 64, 128, T.GemmWarpPolicy.Square),
        (64, 80, 128, T.GemmWarpPolicy.Square),
        (64, 48, 192, T.GemmWarpPolicy.Square),
        (64, 48, 64, T.GemmWarpPolicy.Square),
        (64, 64, 128, T.GemmWarpPolicy.Square),
    ],
)
def test_wmma_gemm_writes_every_output(m, n, threads, policy, transpose_b):
    target = determine_target("hip", return_object=True)
    if not target_is_rdna(target):
        pytest.skip("Requires an RDNA GPU for WMMA execution")

    kernel = tilelang.compile(_gemm_kernel(m, n, threads, policy, transpose_b), target=target)
    rows = torch.arange(1, m + 1, dtype=torch.float16, device="cuda")
    columns = torch.arange(1, n + 1, dtype=torch.float16, device="cuda")
    a = rows[:, None].expand(m, 16).contiguous()
    b = columns[:, None].expand(n, 16).contiguous() if transpose_b else columns[None, :].expand(16, n).contiguous()
    c = torch.full((m, n), -12345, dtype=torch.float32, device="cuda")
    kernel(a, b, c)
    expected = 16 * rows.float()[:, None] * columns.float()[None, :]
    torch.testing.assert_close(c, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
