#ifndef UTILS_GRAPG_MATCH_H
#define UTILS_GRAPG_MATCH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
 public:
  explicit PatternDescribtor(OwningOpRef<ModuleOp> m,
                             llvm::StringRef dialect_namespace)
      : module(std::move(m)) {
    auto entry_bb = (*module->getOps<func::FuncOp>().begin()).begin();
    for (auto it : llvm::enumerate(entry_bb->getArguments())) {
      inputs_map.try_emplace(it.value(), it.index());
    }
    for (auto it : llvm::enumerate(entry_bb->getTerminator()->getOperands())) {
      outputs_map.try_emplace(it.value(), it.index());
    }
    auto operatrion_name =
        entry_bb->getTerminator()->getOperand(0).getDefiningOp()->getName();
    auto root_name = operatrion_name.getStringRef().str();
    root_name_ = dialect_namespace.str() + "." + root_name;
  }

  PatternDescribtor(const PatternDescribtor& other) = delete;
  PatternDescribtor(PatternDescribtor&& other) = default;

  PatternDescribtor& operator=(const PatternDescribtor& other) = delete;
  PatternDescribtor& operator=(PatternDescribtor&& other) = default;
  StringRef getRootName() { return root_name_; }
  PatternBenefit getBenefit() { return 0; }
  ArrayRef<StringRef> getGeneratedNames() { return {}; }

  Value getRootValue() {
    auto entry_bb = (*module->getOps<func::FuncOp>().begin()).begin();
    return entry_bb->getTerminator()->getOperand(0);
  }

  size_t indexOfInput(Value input) const {
    auto it = inputs_map.find(input);
    return it->second;
  }
  size_t indexOfOutput(Value output) {
    auto it = outputs_map.find(output);
    return it->second;
  }

  llvm::DenseMap<Value, size_t>& inputsMap() { return inputs_map; }
  llvm::DenseMap<Value, size_t>& outputsMap() { return outputs_map; }

 private:
  OwningOpRef<ModuleOp> module;
  llvm::DenseMap<Value, size_t> inputs_map;
  llvm::DenseMap<Value, size_t> outputs_map;
  std::string root_name_;
};

class TextFusionRewritePattern : public RewritePattern {
 public:
  using RewritePattern::RewritePattern;

  TextFusionRewritePattern(PatternDescribtor* pattern_describtor,
                           MLIRContext* context,
                           ArrayRef<StringRef> generatedNames = {})
      : RewritePattern(pattern_describtor->getRootName(),
                       pattern_describtor->getBenefit(), context,
                       generatedNames),
        describtor_(pattern_describtor) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;

 private:
  PatternDescribtor* describtor_;
};

class PatternDescribtorManager {
 private:
  PatternDescribtorManager() = default;

  void InitWithContext(MLIRContext* context);

 public:
 // TODO: register dialect
  static PatternDescribtorManager& instance(
      MLIRContext* context = nullptr, llvm::StringRef dialect_namespace = {});

  void registerPatternDescribtor(const std::string& str);

  void registerCommutativeOps(llvm::ArrayRef<llvm::StringRef> names) {
    for (auto it : names) {
      commutative_names_.insert(it);
    }
  }

  bool isCommutativeOp(llvm::StringRef name) {
    return commutative_names_.contains(name);
  }

  auto begin() { return pattern_describtors_.begin(); }
  auto end() { return pattern_describtors_.end(); }

 private:
  MLIRContext* context_ = nullptr;
  std::vector<std::unique_ptr<PatternDescribtor>> pattern_describtors_;
  std::vector<std::string> string_patterns_;
  llvm::StringRef dialect_namespace_;
  llvm::DenseSet<llvm::StringRef> commutative_names_;
};

}  // namespace utils

}  // namespace mlir

#endif