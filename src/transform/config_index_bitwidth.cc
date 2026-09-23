#include "../../3rdparty/tvm/src/tirx/ir/data_type_rewriter.h"
#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "common/int64_promoter.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using namespace arith;

namespace {

// Widen address arithmetic at its use site, without remapping declarations or
// changing the semantics of fixed-width values used to compute an address.
class ForcedIndexInt64Promoter : public DataTypeLegalizer {
public:
  using Parent = DataTypeLegalizer;

private:
  using Parent::VisitExpr_;

  static PrimExpr WidenValue_(const PrimExpr &value) {
    DataType dtype = value.dtype();
    if (dtype.is_int() && dtype.bits() < 64) {
      return cast(dtype.with_bits(64), value);
    }
    return value;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    return WidenValue_(Parent::VisitExpr_(op));
  }

  PrimExpr VisitExpr_(const IntImmNode *op) final {
    if (op->dtype.is_int() && op->dtype.bits() < 64) {
      return IntImm(op->dtype.with_bits(64), op->value);
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const CastNode *op) final {
    return WidenValue_(GetRef<PrimExpr>(op));
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    // IndexLegalizer has already visited this load's address. Its element type
    // and value must retain the buffer's declared dtype.
    return WidenValue_(GetRef<PrimExpr>(op));
  }

  PrimExpr VisitExpr_(const LetNode *op) final {
    // Preserve the binding's value and dtype, just as for a statement Bind.
    return Let(op->var, op->value, VisitExpr(op->body), op->span);
  }

  PrimExpr VisitExpr_(const SelectNode *op) final {
    return Select(op->condition, VisitExpr(op->true_value),
                  VisitExpr(op->false_value), op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      PrimExpr true_value = VisitExpr(op->args[1]);
      PrimExpr false_value = VisitExpr(op->args[2]);
      return Call(true_value.dtype(), op->op,
                  {op->args[0], true_value, false_value}, op->annotations,
                  op->span);
    }
    if (op->op.same_as(builtin::shift_left()) ||
        op->op.same_as(builtin::shift_right()) ||
        op->op.same_as(builtin::bitwise_and()) ||
        op->op.same_as(builtin::bitwise_or()) ||
        op->op.same_as(builtin::bitwise_xor())) {
      PrimExpr value = Parent::VisitExpr_(op);
      if (const auto *call = value.as<CallNode>()) {
        return Call(call->dtype, call->op, call->args, op->annotations,
                    op->span);
      }
      return value;
    }
    // Other intrinsics and external calls have their own argument/result ABI.
    return WidenValue_(GetRef<PrimExpr>(op));
  }
};

class IndexLegalizer : public IRMutatorWithAnalyzer {

public:
  static Stmt Rewrite(const Stmt &stmt, bool force_int64) {
    Analyzer ana;
    auto pass = IndexLegalizer(&ana, force_int64);
    return pass.VisitStmt(stmt);
  }

private:
  IndexLegalizer(arith::Analyzer *ana, bool force_int64)
      : IRMutatorWithAnalyzer(ana), force_int64_(force_int64) {}

  PrimExpr RewriteIndex_(const PrimExpr &index) {
    if (!index->dtype.is_int()) {
      return index;
    }
    if (force_int64_) {
      ForcedIndexInt64Promoter promoter;
      return promoter(index);
    }
    if (index->dtype.bits() < 64) {
      auto int_bound = analyzer_->const_int_bound(index);
      if (int_bound->max_value >= (1LL << (index->dtype.bits() - 1)) - 1 ||
          int_bound->min_value < -(1LL << (index->dtype.bits() - 1))) {
        Int64Promoter promoter;
        return promoter(index);
      }
    }
    return index;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto buffer_store =
        Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    buffer_store.CopyOnWrite()->indices = buffer_store->indices.Map(
        [this](const PrimExpr &index) { return RewriteIndex_(index); });
    return std::move(buffer_store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto buffer_load =
        Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    buffer_load.CopyOnWrite()->indices = buffer_load->indices.Map(
        [this](const PrimExpr &index) { return RewriteIndex_(index); });
    return std::move(buffer_load);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = IRMutatorWithAnalyzer::VisitExpr_(op);
    if (force_int64_ && op->op.same_as(builtin::tvm_access_ptr())) {
      Call call = Downcast<Call>(expr);
      call.CopyOnWrite()->args.Set(2, RewriteIndex_(call->args[2]));
      return call;
    }
    return expr;
  }

  bool force_int64_;
};

} // namespace

tvm::transform::Pass ConfigIndexBitwidth() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Get pass config `tl.config_index_bitwidth`
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Integer> opt_config_index_bitwidth =
        ctxt->GetConfig(kConfigIndexBitwidth, Optional<Integer>());
    bool force_int64 = false;
    if (opt_config_index_bitwidth.defined()) {
      int64_t config_index_bitwidth = opt_config_index_bitwidth.value()->value;
      CHECK(config_index_bitwidth == 32 || config_index_bitwidth == 64,
            ValueError)
          << "tl.config_index_bitwidth must be 32 or 64, but got "
          << config_index_bitwidth;
      force_int64 = config_index_bitwidth == 64;
    }
    f.CopyOnWrite()->body = IndexLegalizer::Rewrite(f->body, force_int64);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ConfigIndexBitwidth", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.ConfigIndexBitwidth",
                        ConfigIndexBitwidth);
}

} // namespace tl
} // namespace tvm
