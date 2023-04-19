#include "include/PassDetial.h"
#include "include/graph_matcher.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir {

namespace {

class DenseCSEPass : public DenseCSEPassBase<DenseCSEPass> {
 public:
  using DenseCSEPassBase::DenseCSEPassBase;
  void runOnOperation() override {
  }
};

}  // namespace

std::unique_ptr<Pass> createDenseCSEPass() {
  return std::make_unique<DenseCSEPass>();
}

}  // namespace mlir
