#include "include/PassDetial.h"
#include "include/graph_matcher.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir {

namespace {

constexpr static llvm::StringRef CommutativeOpNames[] = {"add", "mul"};

class FusionPass : public FusionPassBase<FusionPass> {
 public:
  using FusionPassBase::FusionPassBase;

  

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation()->getContext());
    {
      auto& pattern_describtor = utils::PatternDescribtorManager::instance(
          getOperation()->getContext(),
          mlir::tosa::TosaDialect::getDialectNamespace());

      utils::PatternDescribtorManager::instance().registerCommutativeOps(
          CommutativeOpNames);
      for (auto& describtor : pattern_describtor) {
        patterns.add(std::make_unique<utils::TextFusionRewritePattern>(
            describtor.get(), getOperation()->getContext()));
      }
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace mlir
