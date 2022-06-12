#ifndef UTILS_TRANSFORMS_PASSES_H
#define UTILS_TRANSFORMS_PASSES_H

#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
std::unique_ptr<Pass> createFusionPass();
}

#endif // UTILS_TRANSFORMS_PASSES_H