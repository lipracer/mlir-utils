#include "include/graph_matcher.h"
#include "mlir/Parser/Parser.h"

namespace mlir {

namespace utils {

void PatternDescribtorManager::InitWithContext(MLIRContext* context) {
  if (context == context_) {
    return;
  }
  context_ = context;
  pattern_describtors_.clear();
  for (const auto& str : string_patterns_) {
    auto module = parseSourceString<ModuleOp>(str, context);
    module->dump();
    pattern_describtors_.emplace_back(
        std::make_unique<PatternDescribtor>(std::move(module)));
  }
}

PatternDescribtorManager& PatternDescribtorManager::instance(
    MLIRContext* context) {
  static PatternDescribtorManager __instance;
  __instance.InitWithContext(context);
  return __instance;
}

void PatternDescribtorManager::registerPatternDescribtor(
    const std::string& str) {
  string_patterns_.emplace_back(str);
}

struct TextPatternRegistiter {
  TextPatternRegistiter(const std::string& str) {
    PatternDescribtorManager::instance().registerPatternDescribtor(str);
  }
};

static TextPatternRegistiter GeluPattern(R"(

module {
  func @main(%arg0: none, %arg1: none, %arg2: none, %arg3: none, %arg4: none, %arg5: none) -> none {
    %0 = "broadcast_in_dim"(%arg0) : (none) -> none
    %1 = "broadcast_in_dim"(%arg1) : (none) -> none
    %2 = "broadcast_in_dim"(%arg2) : (none) -> none
    %3 = "broadcast_in_dim"(%arg3) : (none) -> none
    %4 = "broadcast_in_dim"(%arg4) : (none) -> none
    %5 = "pow"(%arg5, %2) : (none, none) -> none
    %6 = "mul"(%5, %3) : (none, none) -> none
    %7 = "add"(%arg5, %6) : (none, none) -> none
    %8 = "fusion"(%5, %6, %7) {fusion_name = "ElementWise"} : (none, none, none) -> none
    %9 = "mul"(%7, %4) : (none, none) -> none
    %10 = "tanh"(%9) : (none) -> none
    %11 = "add"(%1, %10) : (none, none) -> none
    %12 = "mul"(%0, %11) : (none, none) -> none
    %13 = "mul"(%arg5, %12) : (none, none) -> none
    "return"(%13) : (none) -> ()
  }
}

)");

}  // namespace utils
}  // namespace mlir