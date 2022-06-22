#include "include/PassDetial.h"
#include "include/graph_matcher.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

namespace {
class FusionPass : public FusionPassBase<FusionPass> {
 public:
  using FusionPassBase::FusionPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation()->getContext());
    auto& pattern_describtor =
        utils::PatternDescribtorManager::instance(getOperation()->getContext());
    for (auto& describtor : pattern_describtor) {
      patterns.add(std::make_unique<utils::TextFusionRewritePattern>(
          describtor->getRootName(), describtor->getBenefit(),
          getOperation()->getContext()));
    }
    auto result =
        applyPatternsAndFoldGreedily(getOperation(), FrozenRewritePatternSet());
    if (result.failed()) {
      getOperation().emitError("FusionPass rewrite error!");
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace mlir
