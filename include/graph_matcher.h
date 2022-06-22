#ifndef UTILS_GRAPG_MATCH_H
#define UTILS_GRAPG_MATCH_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {

namespace utils {

template <typename T>
struct NodeWrapper {
  T* node;
  T* operator->() { return node; }
  auto getOperands() { return node->getOperands(); }
};

class PatternDescribtor {
 private:
  OwningOpRef<ModuleOp> module;

 public:
  explicit PatternDescribtor(OwningOpRef<ModuleOp>&& m)
      : module(std::move(m)) {}

  PatternDescribtor(const PatternDescribtor& other) = delete;
  PatternDescribtor(PatternDescribtor&& other)
      : module(std::move(other.module)) {}

  PatternDescribtor& operator=(const PatternDescribtor& other) = delete;
  PatternDescribtor& operator=(PatternDescribtor&& other) {
    if (this == &other) return *this;
    module = std::move(other.module);
    return *this;
  }
  StringRef getRootName() { return ""; }
  PatternBenefit getBenefit() { return 0; }
  ArrayRef<StringRef> getGeneratedNames() { return {}; }
};

class TextFusionRewritePattern : public RewritePattern {
 public:
  using RewritePattern::RewritePattern;

  TextFusionRewritePattern(StringRef rootName, PatternBenefit benefit,
                           MLIRContext* context,
                           ArrayRef<StringRef> generatedNames = {})
      : RewritePattern(rootName, benefit, context, generatedNames) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {}
};

class PatternDescribtorManager {
 private:
  PatternDescribtorManager() = default;

  void InitWithContext(MLIRContext* context);

 public:
  static PatternDescribtorManager& instance(MLIRContext* context = nullptr);

  void registerPatternDescribtor(const std::string& str);

  auto begin() { return pattern_describtors_.begin(); }
  auto end() { return pattern_describtors_.end(); }

 private:
  MLIRContext* context_ = nullptr;
  std::vector<std::unique_ptr<PatternDescribtor>> pattern_describtors_;
  std::vector<std::string> string_patterns_;
};

}  // namespace utils

}  // namespace mlir

#endif