/*!
 * \file tl/rocm/op/gemm.cc
 * \brief ROCm implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "rocm/target_utils.h"

#include <cmath>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;

namespace rocm {

namespace {

constexpr const char *kROCmMFMA = "rocm.mfma";
constexpr const char *kROCmWMMA = "rocm.wmma";

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 16;

  // Also guard callers that do not go through ComputeWarpPartition.
  ICHECK_GT(num_warps, 0) << "num_warps must be positive, but got "
                          << num_warps;
  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  // Both the warp partition and each warp's instruction tiles must cover
  // complete rows and columns; otherwise the emitter truncates the remainder.
  auto is_valid = [&](int m, int n) {
    return m * n == num_warps && M >= m * kMPerWarp && N >= n * kNPerWarp &&
           M % (m * kMPerWarp) == 0 && N % (n * kNPerWarp) == 0;
  };

  bool found = false;
  if (policy.IsFullRow()) {
    for (int m = num_warps; m >= 1; m--) {
      if (num_warps % m != 0 || !is_valid(m, num_warps / m))
        continue;
      m_warp = m;
      n_warp = num_warps / m;
      found = true;
      break;
    }
  } else if (policy.IsFullCol()) {
    for (int n = num_warps; n >= 1; n--) {
      if (num_warps % n != 0 || !is_valid(num_warps / n, n))
        continue;
      n_warp = n;
      m_warp = num_warps / n;
      found = true;
      break;
    }
  } else if (policy.IsSquare()) {
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= num_warps; m++) {
      if (num_warps % m != 0)
        continue;
      int n = num_warps / m;
      if (!is_valid(m, n))
        continue;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        m_warp = m;
        n_warp = n;
        found = true;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  if (!found) {
    LOG(FATAL) << "No valid warp partition for ROCm T.gemm: M=" << M
               << ", N=" << N << " cannot be evenly covered by " << num_warps
               << " warps (policy="
               << (policy.IsFullRow()   ? "FullRow"
                   : policy.IsFullCol() ? "FullCol"
                                        : "Square")
               << "). Each warp must own a multiple of " << kMPerWarp
               << " rows and " << kNPerWarp
               << " columns; adjust `threads` or the block tile shape.";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct Gemm {
  static ffi::String SelectInst(const GemmNode &op, int block_size,
                                Target target) {
    (void)op;
    (void)block_size;

    if (TargetIsCDNA(target)) {
      return kROCmMFMA;
    }
    if (TargetIsRDNA(target)) {
      return kROCmWMMA;
    }
    LOG(FATAL) << "Unsupported ROCm target for gemm: " << target->str();
    return kROCmMFMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, ffi::String gemm_inst) {
    (void)gemm_inst;
    const int warp_size = TargetRocmGetWarpSize(target);
    // A block narrower than one wavefront yields zero warps, which used to
    // reach ComputeDefaultWarpPartition and abort the process with SIGFPE on
    // its `M % (m_warp * kMPerWarp)`. Thread counts tuned for NVIDIA's 32-lane
    // warp hit this on CDNA, where the wavefront is 64 wide.
    ICHECK_GE(block_size, warp_size)
        << "T.gemm needs at least one full wavefront, but this block has only "
        << block_size << " threads while the wavefront size for "
        << target->str() << " is " << warp_size
        << ". Raise the kernel's thread count to a multiple of " << warp_size
        << ".";
    int num_warps = block_size / warp_size;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps);
  }

  static bool ReuseExistingSharedLayout(ffi::String gemm_inst) {
    (void)gemm_inst;
    return false;
  }
};

} // namespace rocm

namespace {

bool MatchROCmGemmTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmGemm() {
  RegisterGemmImpl(GemmImpl{
      "rocm.Gemm",
      MatchROCmGemmTarget,
      rocm::Gemm::SelectInst,
      rocm::Gemm::ComputeWarpPartition,
      rocm::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool rocm_gemm_registered = RegisterROCmGemm();

} // namespace

} // namespace tl
} // namespace tvm
