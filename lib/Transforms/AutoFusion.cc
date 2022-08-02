// =============================================================================
//
// Copyright 2019-2021 Enflame. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <deque>
#include <iostream>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Analysis/Dominance.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/*! \brief operator pattern used in graph fusion */
enum class OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // Broadcasting operator, can always map output axis to the input in order.
  // for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
  // Note that the axis need to be in order so transpose is not a bcast
  // operator.
  kBroadcast = 1,
  // Injective operator, can always injectively map output axis to a single
  // input axis.
  // All injective operator can still be safely fused to injective and
  // reduction.
  kInjective = 2,
  // Communicative reduction operator.
  kCommReduce = 3,
  // Complex operation, can still fuse elemwise operations into its output.
  // but cannot chain another complex op
  kOutEWiseFusable = 4,
  // The pattern for tuple nodes. Can fuse into subsequent injective ops,
  // but treated specially
  kTuple = 7,
  // Opaque operation, cannot fuse anything.
  kOpaque = 8
};

OpPatternKind getOpPatternKind(mlir::Operation* op) {
  return mlir::TypeSwitch<mlir::Operation*, OpPatternKind>(op)
      .Case<ConvOp>([](auto op) {})
      .Default([](auto op) {});
}

namespace {
auto getUseEdges(mlir::Value node) {
  llvm::SmallVector<mlir::Value, 4> result;
  for (auto op : node.getUsers()) {
    if (op->getNumResults()) {
      result.push_back(op->getResult(0));
    }
  }
  return result;
}
auto getOperandEdges(mlir::Value node) {
  llvm::SmallVector<mlir::Value, 4> operands;
  if (auto op = node.getDefiningOp()) {
    operands.assign(op->operand_begin(), op->operand_end());
  }
  return operands;
}
static mlir::Value mapOp2Value(mlir::Operation& op) { return op.getResult(0); }

void dumpValue(mlir::Value it) {
  if (!it) {
    llvm::errs() << "empty\n";
  } else if (auto arg = it.dyn_cast<mlir::BlockArgument>()) {
    llvm::errs() << "arg index:" << arg.getArgNumber() << "\n";
  } else {
    llvm::errs() << it << "\n";
  }
}

}  // namespace

template <bool IsPostOrder = false>
struct DominatorTreeTrait {
  static auto getInputs(mlir::Value value) { return getOperandEdges(value); }
  static auto getOutputs(mlir::Value value) { return getUseEdges(value); }

  static auto getRoots(mlir::Block* block) { return block->getArguments(); }
  static bool isLeafNode(mlir::Value value) {
    return llvm::all_of(value.getUsers(),
                        [](auto* op) { return 0 == op->getNumResults(); });
  }
  static auto getContainerRange(mlir::Block* block) {
    return llvm::make_range(llvm::map_iterator(block->begin(), mapOp2Value),
                            llvm::map_iterator(block->end()--, mapOp2Value));
  }
  using RootsType = decltype(std::declval<mlir::Block*>()->getArguments());
};

template <>
struct DominatorTreeTrait<true> {
  static auto getInputs(mlir::Value value) { return getUseEdges(value); }
  static auto getOutputs(mlir::Value value) { return getOperandEdges(value); }

  static auto getRoots(mlir::Block* block) {
    return block->getTerminator()->getOperands();
  }
  static bool isLeafNode(mlir::Value value) {
    return value.isa<mlir::BlockArgument>();
  }
  static auto getContainerRange(mlir::Block* block) {
    llvm::SmallVector<mlir::Value, 4> values;
    for (auto& op : llvm::make_range(++block->rbegin(), block->rend())) {
      values.push_back(mapOp2Value(op));
    }
    for (auto arg : block->getArguments()) {
      values.push_back(arg);
    }
    return values;
  }
  using RootsType =
      decltype(std::declval<mlir::Block*>()->getTerminator()->getOperands());
};

template <typename NodeT, bool IsPostOrder>
class DominatorTree {
  using trait = DominatorTreeTrait<IsPostOrder>;

 public:
  explicit DominatorTree(std::vector<NodeT*> roots) : roots_(trait::getRoots(block)) {
    buildDepthMap(roots_);
    for (auto root : roots_) {
      idomsMap_.insert(std::make_pair(root, mlir::Value()));
    }
    for (auto value : trait::getContainerRange(block)) {
      auto edges = trait::getInputs(value);
      auto begin = std::begin(edges);
      auto end = std::end(edges);
      if (begin == end) {  // const op
        idomsMap_.insert(std::make_pair(value, mlir::Value()));
        continue;
      }
      auto idomNode = *begin++;
      while (begin != end) {
        auto tmp = findCommonAncestor(idomNode, *begin);
        if (depthMap_[tmp] < depthMap_[idomNode]) idomNode = tmp;
        ++begin;
      }
      idomsMap_.insert(std::make_pair(value, idomNode));
    }
  }

  bool dominate(mlir::Value lhs, mlir::Value rhs) {
    if (lhs == rhs) return true;
    return properlyDominate(lhs, rhs);
  }
  bool properlyDominate(mlir::Value lhs, mlir::Value rhs) {
    auto idom = rhs;
    while (idom && idom != lhs) {
      idom = idomsMap_[idom];
    }
    return !!idom;
  }

  mlir::Value findCommonAncestor(mlir::Value lhs, mlir::Value rhs) {
    if (llvm::find(roots_, lhs) != std::end(roots_)) {
      return lhs;
    }
    if (llvm::find(roots_, rhs) != std::end(roots_)) {
      return rhs;
    }
    while (depthMap_[lhs] != depthMap_[rhs]) {
      if (depthMap_[lhs] < depthMap_[rhs]) {
        rhs = idomsMap_[rhs];
      } else if (depthMap_[lhs] > depthMap_[rhs]) {
        lhs = idomsMap_[lhs];
      }
    }
    return lhs;
  }

  size_t getDepth(mlir::Value value) {
    assert(depthMap_.find(value) != depthMap_.end());
    return depthMap_[value];
  }

  mlir::Value getIDomNode(mlir::Value value) {
    auto iter = idomsMap_.find(value);
    if(iter != idomsMap_.end()) {
      return *iter;
    }
    return {};
  }

 private:
  template <typename RangeT>
  void buildDepthMap(RangeT roots, size_t depth = 1) {
    for (auto root : roots) {
      visitValue(root, depth);
    }
  }

  void visitValue(mlir::Value value, size_t& depth) {
    if (depthMap_.find(value) != depthMap_.end()) {
      return;
    }
    depthMap_.insert(std::make_pair(value, depth++));
    if (trait::isLeafNode(value)) {
      return;
    }
    for (auto operand : trait::getOutputs(value)) {
      visitValue(operand, depth);
    }
  }

  llvm::DenseMap<mlir::Value, mlir::Value> idomsMap_;
  llvm::DenseMap<mlir::Value, size_t> depthMap_;
  typename trait::RootsType roots_;
};

namespace {

bool fusable(mlir::Type type) {
  auto tensorType = type.dyn_cast_or_null<mlir::RankedTensorType>();
  if (!tensorType) return false;
  if (!tensorType.hasStaticShape()) return false;
  if (tensorType.getNumElements() != 1) return false;
  return true;
}

bool isElewiseOp(mlir::Operation* op) {
}

}  // namespace

class AutoFusionPass : public mlir::OperationPass<AutoFusionPass> {
  struct ValueWrapper {
    mlir::Value value;
    size_t index{-1};
    bool op{false};
    bool visited{false};
    bool operator<(const ValueWrapper& other) const {
      return this->index < other.index;
    }
    bool operator==(const ValueWrapper& other) const {
      return this->value == other.value;
    }
    ValueWrapper(mlir::Value value, size_t index, bool op = false)
        : value(value), index(index), op(op), visited(false) {}
  };

  bool fusableOp(mlir::Operation* op) {
    return op->getNumResults() && fusable(op->getResult(0).getType()) &&
           isElewiseOp(op);
  }

 public:
  using DomTree = DominatorTree<true>;

  void runFuse(mlir::Block& block, DomTree domTree, size_t phase) {
    for (auto& op : llvm::reverse(block)) {
      auto opPattern = getOpPatternKind(&op);
      auto idom = domTree.getIDomNode(op.getResult(0));
      if (!idom) continue;
      if (opPattern == OpPatternKind::kOutEWiseFusable) {
        if (phase != 0) continue;
        if (idom && dom_node->pattern == OpPatternKind::kElemWise) {

          auto fcond = [](OpPatternKind kind, bool is_sink) {
            return kind <= kBroadcast;
          };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (opPattern <= OpPatternKind::kBroadcast) {
      } else if (opPattern == OpPatternKind::kInjective ||
                 opPattern == OpPatternKind::kTuple) {
      } else {
      }
    }
  }

  void runOnOperation() override {
    auto module_op = llvm::cast<mlir::ModuleOp>(getOperation());
    mlir::FuncOp main_op = module_op.lookupSymbol<mlir::FuncOp>("main");
    DomTree dominatorTree(&main_op.front());



    auto insertOperation = [&](mlir::Operation * op) -> auto{
      auto preIter = unionSet.end();
      for (auto operand : op->getOperands()) {
        auto valueWrap = ValueWrapper(operand, dominatorTree.getDepth(operand));
        auto iter = unionSet.findValue(valueWrap);
        if (iter == unionSet.end()) {
          auto curIter = unionSet.insert(valueWrap);
          if (preIter != unionSet.end()) {
            unionSet.unionSets(preIter->getData(), curIter->getData());
          }
          preIter = curIter;
        } else {
          if (preIter != unionSet.end()) {
            unionSet.unionSets(preIter->getData(), iter->getData());
          }
        }
      }
      auto retIter = unionSet.insert(ValueWrapper(
          op->getResult(0), dominatorTree.getDepth(op->getResult(0)), true));
      unionSet.unionSets(preIter->getData(), retIter->getData());
    };

    for (auto& op : main_op.front()) {
      if (!fusableOp(&op)) {
        continue;
      }
      insertOperation(&op);
    }
    for (auto begin = unionSet.begin(); begin != unionSet.end(); ++begin) {
      auto leaderIter = unionSet.findLeader(begin);
      if (!leaderValues.insert(leaderIter->value).second) {
        continue;
      }
      (void)fuseOps(llvm::make_range(leaderIter, unionSet.member_end()));
    }
  }
  using union_set_iterator =
      llvm::EquivalenceClasses<ValueWrapper>::member_iterator;
  mlir::Operation* fuseOps(llvm::iterator_range<union_set_iterator> range) {
    llvm::SmallVector<mlir::Operation*, 4> ops;
    llvm::SmallVector<mlir::Value, 4> inputs;
    llvm::SmallVector<FusionOutputValueInfo, 4> outputs;
    for (const auto& valueWrap : range) {
      if (valueWrap.op) {
        ops.push_back(valueWrap.value.getDefiningOp());
        outputs.emplace_back(valueWrap.value);
      } else {
        inputs.push_back(valueWrap.value);
      }
    }
    FusionOpCreator creator(ops, inputs, outputs, {}, static_cast<int32_t>(0));
    auto fusion_op = creator.CreateFusionOp();
    if (!fusion_op) {
    } else {
      fusion_op->setAttr(
          kFusionNameAttri,
          mlir::StringAttr::get("ElementWise", fusion_op->getContext()));
    }
    return fusion_op;
  }

 private:
  llvm::EquivalenceClasses<ValueWrapper> unionSet;
  llvm::DenseSet<mlir::Value> leaderValues;
};



}  // namespace mlir
