
#include <deque>
#include <iostream>

#include "dtu/util/switch_logging.h"
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

OpPatternKind operator&(OpPatternKind lhs, OpPatternKind rhs) {
  return std::max(lhs, rhs);
}

OpPatternKind getOpPatternKind(mlir::Operation* op) {
  if (!op) return OpPatternKind::kOpaque;
  return mlir::TypeSwitch<mlir::Operation*, OpPatternKind>(op)
      .Case<ConvOp>([](auto op) { return OpPatternKind::kOutEWiseFusable; })
      .Default([](auto op) {
        if (is_binary_ew(op) || is_unary(op)) {
          return OpPatternKind::kElemWise;
        }
        return OpPatternKind::kOpaque;
      });
}

OpPatternKind getOpPatternKind(mlir::Value value) {
  return getOpPatternKind(value.getDefiningOp());
}

namespace {
auto getUseEdges(mlir::Value node) {
  llvm::SmallVector<mlir::Value, 4> result;
  for (auto& use : node.getUses()) {
    result.insert(result.end(), use.getOwner()->result_begin(),
                  use.getOwner()->result_end());
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
  template <typename RangeT>
  static auto visitPostorderRange(RangeT range) {
    return range;
  }
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
  template <typename RangeT>
  static auto visitPostorderRange(RangeT range) {
    return llvm::reverse(range);
  }
};

template <typename NodeT, bool IsPostOrder>
class DominatorTree {
 public:
  using trait = DominatorTreeTrait<IsPostOrder>;
  class TreeNode {
    NodeT data;
    // depth of dominator tree
    size_t depth;
    // index of postorder
    size_t index;
    // idom node
    TreeNode* parent;
    OpPatternKind pattern;

   public:
    TreeNode* getParent() { return parent; }
    TreeNode* setParent(TreeNode* parent) { this->parent = parent; }

    size_t getDepth() { return depth; }
    void setDepth(size_t depth) { this->depth = depth; }

    size_t getIndex() { return index; }
    void setIndex(size_t index) { this->index = index; }

    NodeT getData() { return data; }
    OpPatternKind getPattern() { return pattern; }
    void setPattern(OpPatternKind kind) { pattern = kind; }

    TreeNode(NodeT data, size_t depth = 0, size_t index = std::string::npos,
             TreeNode* parent = nullptr)
        : data(data), depth(depth), index(index), parent(parent) {}
    TreeNode() : data(), depth(0), index(std::string::npos), parent(nullptr) {
      assert(false && "should not call!");
    }
  };
  template <typename IteratorT>
  explicit DominatorTree(llvm::ArrayRef<NodeT> roots,
                         llvm::iterator_range<IteratorT> postorderRange)
      : roots_(std::begin(roots), std::end(roots)) {
    size_t index = 0;
    for (auto value : trait::visitPostorderRange(postorderRange)) {
      TreeNode* idomNode = nullptr;
      if (!roots_.count(value)) {
        auto edges = trait::getInputs(value);
        auto begin = std::begin(edges);
        auto end = std::end(edges);
        idomNode = findCommonAncestor(edges);
      }
      nodeMap_.try_emplace(value, value,
                           idomNode ? idomNode->getDepth() + 1 : 1, index++,
                           idomNode);
    }
  }

  DominatorTree() = default;

  bool dominate(NodeT lhs, NodeT rhs) {
    if (lhs == rhs) return true;
    return properlyDominate(lhs, rhs);
  }
  bool properlyDominate(NodeT lhs, NodeT rhs) {
    auto idom = &nodeMap_[rhs];
    auto lhsNode = &nodeMap_[lhs];
    while (idom && idom != lhsNode) {
      idom = idom->getParent();
    }
    return !!idom;
  }

  template <typename RangeT>
  TreeNode* findCommonAncestor(RangeT nodes) {
    if (0 == nodes.size()) {
      return nullptr;
    }
    if (1 == nodes.size()) {
      return &nodeMap_[*nodes.begin()];
    }
    auto begin = nodes.begin();
    auto idomNode = &nodeMap_[*begin++];
    while (begin != nodes.end()) {
      auto tmp = findCommonAncestor(idomNode, &nodeMap_[*begin]);
      if (tmp->getDepth() < idomNode->getDepth()) {
        idomNode = tmp;
      }
      ++begin;
    }
    return idomNode;
  }

  TreeNode* findCommonAncestor(TreeNode* lhs, TreeNode* rhs) {
    if (roots_.count(lhs->getData())) return lhs;
    if (roots_.count(rhs->getData())) return rhs;
    while (lhs->getDepth() != rhs->getDepth()) {
      if (lhs->getDepth() < rhs->getDepth()) {
        rhs = rhs->getParent();
      } else if (lhs->getDepth() > rhs->getDepth()) {
        lhs = lhs->getParent();
      }
    }
    return lhs;
  }

  size_t getIndex(mlir::Value value) {
    assert(nodeMap_.find(value) != nodeMap_.end());
    return nodeMap_[value].getIndex();
  }

  TreeNode* getIDomNode(mlir::Value value) {
    auto iter = nodeMap_.find(value);
    if (iter != nodeMap_.end()) {
      return iter->second.getParent();
    }
    return {};
  }

  TreeNode* getTreeNode(mlir::Value value) {
    assert(nodeMap_.find(value) != nodeMap_.end() && "node is not exist!");
    return &nodeMap_[value];
  }

 private:
  llvm::DenseMap<NodeT, TreeNode> nodeMap_;
  llvm::SmallSetVector<NodeT, 4> roots_;
};

namespace {

bool fusableScalar(mlir::Type type) {
  auto tensorType = type.dyn_cast_or_null<mlir::RankedTensorType>();
  if (!tensorType) return false;
  if (!tensorType.hasStaticShape()) return false;
  if (tensorType.getNumElements() != 1) return false;
  return true;
}

// TODO(user): support elementwsie fusion op and unary op
bool isElewiseOp(mlir::Operation* op) { return is_binary_ew(op); }

}  // namespace

class AutoFusionPass : public mlir::OperationPass<AutoFusionPass> {
  struct ValueWrapper {
    mlir::Value value;
    size_t index{std::string::npos};
    bool op{false};
    // postorder
    bool operator<(const ValueWrapper& other) const {
      return this->index > other.index;
    }

    ValueWrapper(mlir::Value value, size_t index, bool op = false)
        : value(value), index(index), op(op) {}
  };

  bool fusableScalarOp(mlir::Operation* op) {
    return op->getNumResults() && fusableScalar(op->getResult(0).getType()) &&
           isElewiseOp(op);
  }

 public:
  using DomTree = DominatorTree<mlir::Value, true>;
  using TreeNode = DomTree::TreeNode;

  void initUnionSet(llvm::ArrayRef<mlir::Value> postorder) {
    for (auto value : postorder) {
      unionSet.insert(ValueWrapper(value, domTree.getIndex(value),
                                   !value.isa<mlir::BlockArgument>()));
    }
  }

  auto findValue(mlir::Value value) {
    return unionSet.findValue(ValueWrapper(value, domTree.getIndex(value)));
  }

  template <typename FT>
  bool checkPath(TreeNode* current, TreeNode* idom, FT pred) {
    llvm::DenseSet<mlir::Value> visitor;
    if (!visitor.insert(current->getData()).second) return true;
    if (current->getData() == idom->getData()) return true;
    if (getOpPatternKind(current->getData().getDefiningOp()) >
        OpPatternKind::kBroadcast) {
      return false;
    }
    for (auto& value : current->getData().getUses()) {
      if (!checkPath(domTree.getTreeNode(value.getOwner()->getResult(0)), idom,
                     pred)) {
        return false;
      }
    }
    return true;
  }

  // union fused ops
  void commitFuse(TreeNode* current, TreeNode* target) {
    llvm::DenseSet<mlir::Value> visitor;
     if (!visitor.insert(current->getData()).second) return;
    // leaf node
    auto currentIter = findValue(current->getData());
    auto targetIter = findValue(current->getData());
    if (getOpPatternKind(current->getData().getDefiningOp()) >
        OpPatternKind::kBroadcast) {
      const_cast<ValueWrapper&>(currentIter->getData()).op = false;
    }
    unionSet.unionSets(currentIter->getData(), targetIter->getData());
    for (auto& value : current->getData().getUses()) {
      commitFuse(domTree.getTreeNode(value.getOwner()->getResult(0)), target);
    }
  }

  void runFuse(llvm::ArrayRef<mlir::Value> postorder, size_t phase) {
    for (auto value : llvm::reverse(postorder)) {
      auto opPattern = getOpPatternKind(value);
      auto treeNode = domTree.getTreeNode(value);
      auto idom = domTree.getIDomNode(value);
      if (!idom) continue;
      if (opPattern == OpPatternKind::kOutEWiseFusable) {
        if (phase != 0) continue;
        if (idom && idom->getPattern() == OpPatternKind::kElemWise) {
          auto fcond = [](OpPatternKind kind, bool is_sink) {
            return kind <= OpPatternKind::kBroadcast;
          };
          if (checkPath(treeNode, idom, fcond)) {
            commitFuse(treeNode, idom);
          }
        }
      } else if (opPattern <= OpPatternKind::kBroadcast) {
      } else if (opPattern == OpPatternKind::kInjective ||
                 opPattern == OpPatternKind::kTuple) {
      } else {
      }
    }
  }

  // reachable map may be fast but too heavy
  bool slowReachableFuseSet(mlir::Operation* op) {
    // auto comonProducer =
    // auto iter = llvm::find_if(op->getOperands(), [](auto operand) {
    //   auto iter = unionSet.findValue(operand);
    //   return iter != unionSet.end();
    // });
    return false;
  }

  void mergeCommonProducerFusion() {
    llvm::SmallVector<mlir::Operation*, 4> scalarOps;
    for (auto begin = unionSet.begin(), end = unionSet.end(); begin != end;
         ++begin) {
      if (std::next(unionSet.member_begin(begin)) != unionSet.member_end()) {
        continue;
      }
      if (!begin->getData().op) {
        continue;
      }
      auto op = begin->getData().value.getDefiningOp();
      if (!fusableScalarOp(op)) {
        continue;
      }
      scalarOps.push_back(op);
    }
    for (auto op : scalarOps) {
      if (slowReachableFuseSet(op)) continue;
      auto mem_iter =
          unionSet.unionSets(findValue(op->getOperand(0))->getData(),
                             findValue(op->getResult(0))->getData());
      for (auto operand : op->getOperands()) {
        auto iter = findValue(operand);
        unionSet.unionSets(iter->getData(), *mem_iter);
        // check cycle
      }
    }
  }

  void runOnOperation() override {
    auto module_op = llvm::cast<mlir::ModuleOp>(getOperation());
    mlir::FuncOp main_op = module_op.lookupSymbol<mlir::FuncOp>("main");
    llvm::SmallVector<mlir::Value, 4> roots;
    for (auto it : main_op.front().getTerminator()->getOperands()) {
      roots.push_back(it);
    }
    llvm::SmallVector<mlir::Value, 4> postorder;
    postorder.assign(main_op.front().args_begin(), main_op.front().args_end());
    auto range = llvm::make_range(std::begin(main_op.front()),
                                  --std::end(main_op.front()));

    for (auto& op : range) {
      for (auto ret : op.getResults()) {
        postorder.push_back(ret);
      }
    }
    domTree = DomTree(
        roots, llvm::make_range(std::begin(postorder), std::end(postorder)));
    initUnionSet(postorder);

    for (size_t phase = 0; phase < 3; ++phase) {
      runFuse(postorder, phase);
    }

    mergeCommonProducerFusion();

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
    if (ops.size() <= 1) return nullptr;
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
  DomTree domTree;
};

}  // namespace mlir

