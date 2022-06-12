

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "PassDetial.h"

namespace mlir {

namespace {
class FusionPass : public FusionPassBase<FusionPass> {
 public:
  using FusionPassBase::FusionPassBase;

  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<Pass> createFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace mlir
