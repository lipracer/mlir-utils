#ifndef UTILS_TRANSFORMS_PASSES_H
#define UTILS_TRANSFORMS_PASSES_H

#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createFusionPass();

namespace utils {
#define GEN_PASS_REGISTRATION
#include "include/Transforms/utils_passes.h.inc"
inline void registerAllUTILSPasses() { registerUTILSPasses(); }
}  // namespace utils

}  // namespace mlir

#endif // UTILS_TRANSFORMS_PASSES_H