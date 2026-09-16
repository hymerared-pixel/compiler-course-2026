#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct MulAddToFmaPattern : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                 PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    auto tryFuse = [&](Value mulCandidate,
                        Value addend) -> arith::MulFOp {
      auto mulOp = mulCandidate.getDefiningOp<arith::MulFOp>();
      if (!mulOp)
        return nullptr;

      if (!mulOp.getResult().hasOneUse())
        return nullptr;

      if (mulOp.getType() != addOp.getType() ||
          addend.getType() != addOp.getType())
        return nullptr;

      return mulOp;
    };

    Value a, b, c;
    if (arith::MulFOp mulOp = tryFuse(lhs, rhs)) {
      a = mulOp.getLhs();
      b = mulOp.getRhs();
      c = rhs;
    } else if (arith::MulFOp mulOp = tryFuse(rhs, lhs)) {
      a = mulOp.getLhs();
      b = mulOp.getRhs();
      c = lhs;
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<math::FmaOp>(addOp, addOp.getType(), a, b, c);
    return success();
  }
};

struct FmaFusionPass
    : public PassWrapper<FmaFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FmaFusionPass)

  StringRef getArgument() const final { return "fma-fusion"; }
  StringRef getDescription() const final {
    return "Fuse arith.mulf + arith.addf chains into math.fma";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<math::MathDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MulAddToFmaPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFmaFusionPass() {
  return std::make_unique<FmaFusionPass>();
}

static PassRegistration<FmaFusionPass> pass;
