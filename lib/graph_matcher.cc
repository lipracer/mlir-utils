#include <deque>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>

#include "llvm/ADT/SetVector.h"



namespace {
template <typename T, typename Pred>
T moveToEnd(T first, T last, Pred pred) {
  auto cur = first--;
  while (cur < last) {
    // nonone op
    if (!pred(*cur)) {
      std::iter_swap(++first, cur);
    }
    ++cur;
  }
  return ++first;
}

struct ValueWrapper {
  mlir::Value value;
  size_t order;
  ValueWrapper(mlir::Value v, size_t o) : value(v), order(o) {}
  bool operator<(const ValueWrapper& other) const {
    return order < other.order;
  }
};


// it is binary tree, we just recursion collect
// leaf node is not commutative and not null
void getBinaryTreeLeaf(void* class_id, mlir::Operation* root,
                       llvm::SmallVectorImpl<mlir::Value>& leafNodes,
                       std::function<void(mlir::Operation*)> callback = {}) {
  if (!root) return;
  if (callback) callback(root);
  for (auto operand : root->getOperands()) {
    auto operand_op = operand.getDefiningOp();
    if (!operand_op) {
      leafNodes.push_back(operand);
      continue;
    }
    auto id = operand_op->getAbstractOperation()
                  ? operand_op->getAbstractOperation()->classID
                  : nullptr;
    if (id != class_id) {
      leafNodes.push_back(operand);
      continue;
    }
    // only Commutative reach here
    getBinaryTreeLeaf(class_id, operand.getDefiningOp(), leafNodes, callback);
  }
}

// if value is argument or defin op is mlir standard op,
// it is difficulty to sort
struct ValueCanonicalizeLess {
  static bool ingore(mlir::Value v) {
    return !v.getDefiningOp();
  }
  // descending sort
  // E.g add(var,dot) match add(const,dot)
  //         |                   |
  //     add(dot,var)       add(dot,const)
  bool operator()(mlir::Value lhs, mlir::Value rhs) const {
    return lhs.getDefiningOp()->getAbstractOperation()->classID <
           rhs.getDefiningOp()->getAbstractOperation()->classID;
  }
};

inline bool same_type_op(mlir::Operation* lhs, mlir::Operation* rhs) {
  auto lhs_abs = lhs->getAbstractOperation();
  auto rhs_abs = rhs->getAbstractOperation();
  if (lhs_abs && rhs_abs) {
    return lhs_abs->classID == rhs_abs->classID;
  }
  return lhs->getName() == rhs->getName();
}

inline bool same_type(mlir::Value lhs, mlir::Value rhs) {
  return same_type_op(lhs.getDefiningOp(), rhs.getDefiningOp());
}

// find all of adjacent rangs and move them to end, because we can not judge
// witch one of them is matched, but adjacent rang0 is different with other
// it may be accelerate match. someone can improve its.
template <typename V, typename Iterator = typename V::iterator>
static typename std::tuple<Iterator, Iterator> canonicalizeOperand(V& lhs,
                                                                   V& rhs) {
  auto lhs_invalid_iter =
      moveToEnd(lhs.begin(), lhs.end(), ValueCanonicalizeLess::ingore);
  auto rhs_invalid_iter =
      moveToEnd(rhs.begin(), rhs.end(), ValueCanonicalizeLess::ingore);
  // Commutative operands are often duplicated, and we need to avoid these in
  // order to improve the efficiency of the algorithm
  // The original order is most likely to match
  std::stable_sort(lhs.begin(), lhs_invalid_iter, ValueCanonicalizeLess());
  std::stable_sort(rhs.begin(), rhs_invalid_iter, ValueCanonicalizeLess());
  return std::make_pair(lhs_invalid_iter, rhs_invalid_iter);
}



bool Equivalence(mlir::Operation* lhs, mlir::Operation* rhs) {
  // tuple op abstract is null
  if (!lhs->getAbstractOperation() || !rhs->getAbstractOperation()) {
    return lhs->getName().getStringRef() == rhs->getName().getStringRef();
  }
  if (lhs->getAbstractOperation()->classID !=
      rhs->getAbstractOperation()->classID) {
    return false;
  }
  return true;
}

bool Equivalence(mlir::Value lhs, mlir::Value rhs) {
  // tuple op abstract is null
  return Equivalence(lhs.getDefiningOp(), rhs.getDefiningOp());
}

bool isCommutative(mlir::Operation* op) {
  return op->hasTrait<mlir::OpTrait::IsCommutative>();
}

using value_pair = std::pair<mlir::Value, mlir::Value>;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, value_pair pair) {
  if (auto arg = pair.first.template dyn_cast<mlir::BlockArgument>()) {
    os << "<block argument>" << std::to_string(arg.getArgNumber());
  } else {
    os << pair.first;
  }
  os << "\n" << pair.second;
  if (auto arg = pair.second.template dyn_cast<mlir::BlockArgument>()) {
    os << "<block argument>" << std::to_string(arg.getArgNumber());
  } else {
    os << pair.second;
  }
  return os;
}

class empty_ostream : public llvm::raw_ostream {
  void write_impl(const char* Ptr, size_t Size) final {}
  uint64_t current_pos() const final { return 0; }
  empty_ostream() : llvm::raw_ostream(true) {}
  ~empty_ostream() final {}

 public:
  static llvm::raw_ostream& instance() {
    static empty_ostream ostream;
    return ostream;
  }
};

// traverse all of text pattern grap with BFS order, that always find the
// mismatches in advance.
// if we find commutative root, we will use backtracking algorithm.
// TODO(use) maybe we need DFS and global matched succeed table
// Avoid multiple visits to a node and its lead nodes
struct PatternMatcher {
  llvm::SmallVector<value_pair, 4> roots;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 4> input_values;
  llvm::SmallVector<mlir::Operation*, 4> fused_ops;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 4> output_values;
  llvm::DenseSet<mlir::Value> visited_set;
  PatternMatcher * pre_matcher = nullptr;

  llvm::DenseSet<mlir::Value> fused_ops_value_set;

  llvm::raw_ostream* log_ostream;
  size_t depth;
  bool debug;
  PatternMatcher(llvm::ArrayRef<value_pair> r, size_t d = 0, bool dbg = false,
                 PatternMatcher* pre = nullptr)
      : roots(r.begin(), r.end()),
        log_ostream(nullptr),
        depth(d),
        debug(dbg),
        pre_matcher(pre) {
    if (debug) {
      log_ostream = &llvm::outs();
    } else {
      log_ostream = &empty_ostream::instance();
    }
  }
  PatternMatcher() : PatternMatcher(llvm::ArrayRef<value_pair>()) {}

  PatternMatcher(const PatternMatcher& other) = default;
  PatternMatcher& operator=(const PatternMatcher& other) = default;
  PatternMatcher(PatternMatcher&& other) = default;
  PatternMatcher& operator=(PatternMatcher&& other) = default;
  ~PatternMatcher() {
    TEXT_LOGW(log() << "exit stack---------------" << std::to_string(depth)
                    << "\n")
  }

  void clear() {
    roots.clear();
    input_values.clear();
    fused_ops.clear();
    output_values.clear();
    visited_set.clear();
  }

  void set_root(llvm::ArrayRef<value_pair> r) {
    roots.insert(roots.end(), r.begin(), r.end());
  }

  void set_depth(size_t d) { depth = d; }
  llvm::raw_ostream& log() { return *log_ostream; }

  llvm::raw_ostream& log_fail() {
    log() << "fail:";
    return log();
  }
  void enable_debug(bool dbg = true) { debug = dbg; }
  llvm::raw_ostream& log_fail_node(value_pair n) {
    log() << "fail:" << n << "\n";
    return log();
  }

  llvm::raw_ostream& log_success_node(value_pair n) {
    log() << "success:\n" << n << "\n";
    return log();
  }

  void merge(const PatternMatcher& other) {
    input_values.insert(input_values.end(), other.input_values.begin(),
                        other.input_values.end());
    insert_fused_ops(other.fused_ops.begin(), other.fused_ops.end());
    output_values.insert(output_values.end(), other.output_values.begin(),
                         other.output_values.end());
    for (auto it : other.visited_set) visited_set.insert(it);
  }

  void push_back_fused_ops(mlir::Operation* op) {
    fused_ops_value_set.insert(op->getResult(0));
    fused_ops.push_back(op);
  }
  template <typename T>
  void insert_fused_ops(T first, T last) {
    for (auto it = first; it != last; ++it) {
      fused_ops_value_set.insert((*it)->getResult(0));
    }
    fused_ops.insert(fused_ops.end(), first, last);
  }

  template <typename T>
  auto sameTypeRangeMatch(T& lhs, T& rhs, PatternMatcher& matcher) {
    // The original order is generated by the same algorithm, so they are more
    // likely to match, so we use this order to permute
    llvm::SmallVector<
        std::pair<typename std::decay<decltype(rhs[0])>::type, size_t>, 16>
        wrapper;
    for (size_t i = 0; i < rhs.size(); ++i) {
      wrapper.emplace_back(rhs[i], i);
    }
    bool matched = false;
    // TODO(leo.chen): If the match is successful, we need to narrow the
    // permutation interval
    do {
      matcher.clear();
      for (auto it : llvm::enumerate(llvm::zip(lhs, wrapper))) {
        matcher.roots.emplace_back(std::get<0>(it.value()),
                                   std::get<1>(it.value()).first);
      }
      if (matcher()) {
        return true;
      }
      // match fail we need clear state
      matcher.clear();
    } while (!matched &&
             std::next_permutation(std::begin(wrapper), std::end(wrapper),
                                   [](auto lhsV, auto rhsV) {
                                     return lhsV.second < rhsV.second;
                                   }));
    if (matched) {
      std::transform(std::begin(wrapper), std::end(wrapper), rhs.begin(),
                     [](auto w) { return w.first; });
    }
    return matched;
  }

  bool visited(mlir::Value node) {
    auto cur = this;
    while (cur) {
      if (cur->fused_ops_value_set.find(node) !=
          cur->fused_ops_value_set.end()) {
        return true;
      }
      cur = cur->pre_matcher;
    }

    if (this->visited_set.find(node) != this->visited_set.end()) {
      return true;
    }
    return false;
  }

  bool matchAlreadMatchedInputs(value_pair node) {
    auto cur = this;
    while (cur) {
      auto iter = llvm::find_if(
          cur->input_values, [&](auto it) { return node.first == it.first; });
      if (iter != cur->input_values.end()) {
        if (node.second != iter->second) {
          return false;
        }
      }
      cur = cur->pre_matcher;
    }
    return true;
  }

  bool operator()() {
    TEXT_LOGW(log() << "entry stack---------------" << std::to_string(depth)
                    << "\n")
    std::deque<value_pair> queue(roots.begin(), roots.end());
    while (!queue.empty()) {
      auto cur = queue.front();
      queue.pop_front();
      TEXT_LOGW(log() << "try match:" << cur << "\n")
      if (cur.first.template isa<mlir::BlockArgument>()) {
        if (!matchAlreadMatchedInputs(cur)) {
          TEXT_LOGW(log() << "fail matchAlreadMatchedInputs\n" << cur << "\n")
          return false;
        }
        input_values.push_back(cur);
        TEXT_LOGW(log() << "argument match:")
        TEXT_LOGW(log_success_node(cur))
        continue;
      }
      if (visited(cur.first)) {
        continue;
      }
      visited_set.insert(cur.first);
      if (cur.second.template isa<mlir::BlockArgument>()) {
        TEXT_LOGW(log_fail_node(cur))
        return false;
      }
      if (!Equivalence(cur.first, cur.second)) {
        TEXT_LOGW(log_fail_node(cur))
        return false;
      } else {
        push_back_fused_ops(cur.second.getDefiningOp());
        TEXT_LOGW(log_success_node(cur))
      }
      if (llvm::any_of(cur.first.getUses(), [](const mlir::OpOperand& op) {
            return op.getOwner() == op.getOwner()->getBlock()->getTerminator();
          })) {
        output_values.push_back(cur);
      }
      llvm::SmallVector<mlir::Value, 16> lhs_operands;
      llvm::SmallVector<mlir::Value, 16> rhs_operands;
      if (isCommutative(cur.first.getDefiningOp())) {
        getBinaryTreeLeaf(
            cur.first.getDefiningOp()->getAbstractOperation()->classID,
            cur.first.getDefiningOp(), lhs_operands);
        llvm::SmallVector<mlir::Operation*, 4> commutative_ops;
        getBinaryTreeLeaf(
            cur.second.getDefiningOp()->getAbstractOperation()->classID,
            cur.second.getDefiningOp(), rhs_operands,
            [&commutative_ops](mlir::Operation* op) -> void {
              commutative_ops.push_back(op);
            });

        size_t origin_operand_size = lhs_operands.size();
        // if there only one comutative op, we just visit the root node
        if (lhs_operands.size() == 2) {
          commutative_ops.resize(1);
          rhs_operands.assign(commutative_ops[0]->operand_begin(),
                              commutative_ops[0]->operand_end());
        }
        if (lhs_operands.size() > rhs_operands.size()) {
          TEXT_LOGW(log() << "fail lhs_operands vs rhs_operands size:"
                          << lhs_operands.size() << " vs "
                          << rhs_operands.size() << "\n")
          return false;
        }
        llvm::SmallVector<mlir::Value, 16>::iterator lhs_iter, rhs_iter;
        std::tie(lhs_iter, rhs_iter) =
            canonicalizeOperand(lhs_operands, rhs_operands);
        TEXT_LOGW(log() << "debug after sort lhs:\n")
        TEXT_LOGW(for (auto lhs : lhs_operands) log() << lhs << "\n")
        TEXT_LOGW(log() << "debug rhs:\n")
        TEXT_LOGW(for (auto rhs : rhs_operands) log() << rhs << "\n")
        // both lhs and rhs sortable operands's size is not equal
        // match fail

        // if all of lhs and all of rhs can sort. we need no to permutate
        // those,we can insert those front of queue and immediate match
        bool has_same_type = std::adjacent_find(lhs_operands.begin(), lhs_iter,
                                                same_type) != lhs_iter;
        if (has_same_type || lhs_iter != lhs_operands.end() ||
            rhs_iter != rhs_operands.end()) {
          llvm::MutableArrayRef<mlir::Value> lhs_sorted_operands(
              lhs_operands.begin(), lhs_iter);
          llvm::MutableArrayRef<mlir::Value> rhs_sorted_operands(
              rhs_operands.begin(), rhs_iter);

          llvm::MutableArrayRef<mlir::Value> lhs_unsorted_operands(
              lhs_iter, lhs_operands.end());
          llvm::MutableArrayRef<mlir::Value> rhs_unsorted_operands(
              rhs_iter, rhs_operands.end());

          // drawer principle
          if (lhs_sorted_operands.size() > rhs_sorted_operands.size()) {
            TEXT_LOGW(log() << "fail sorted_operands size:"
                            << lhs_sorted_operands.size() << " vs "
                            << rhs_sorted_operands.size() << "\n")
            return false;
          }

          // we need save matched operands then we can find witch
          // commutative_ops is matched
          llvm::SmallVector<mlir::Value, 4> matched_operands;

          // both lhs and rhs unable sort operands may be not  equal, eg.
          //                mul0
          //                 \
          //   mul     mul   mul    mul1
          //    \      /      \     /
          //       mul          mul
          // we can not judge mul0 or mul1 may be matched, so we need permutate
          // all of thems and try match
          llvm::SmallVector<PatternMatcher, 4> sub_patterns;
          auto lower_iter = lhs_sorted_operands.begin();
          while (lower_iter != lhs_sorted_operands.end()) {
            auto range = std::equal_range(rhs_sorted_operands.begin(),
                                          rhs_sorted_operands.end(),
                                          *lower_iter, ValueCanonicalizeLess());
            // can not find OpType in rhs range
            if (range.first == rhs_sorted_operands.end()) {
              TEXT_LOGW(log_fail() << "not find same type!")
              return false;
            }
            auto upper_iter =
                std::upper_bound(lower_iter, lhs_sorted_operands.end(),
                                 *lower_iter, ValueCanonicalizeLess());
            auto lhs_range = llvm::makeMutableArrayRef(
                lower_iter, std::distance(lower_iter, upper_iter));
            auto rhs_range = llvm::makeMutableArrayRef(
                range.first, std::distance(range.first, range.second));
            if (lhs_range.size() > rhs_range.size()) {
              TEXT_LOGW(log_fail() << "same type mismatch:" << lhs_range.size()
                                   << " vs " << rhs_range.size())
              return false;
            }
            sub_patterns.emplace_back(llvm::ArrayRef<value_pair>(), depth + 1,
                                      debug, this);

            if (!sameTypeRangeMatch(lhs_range, rhs_range,
                                    sub_patterns.back())) {
              TEXT_LOGW(log_fail() << "sameTypeRangeMatch mismatch!")
              return false;
            }
            lower_iter = upper_iter;
          }
          for (auto& pattern : sub_patterns) {
            for (auto value : pattern.roots) {
              matched_operands.push_back(value.second);
            }
            merge(pattern);
          }
          sub_patterns.clear();
          llvm::SmallVector<mlir::Value, 16> rhs_operands_backup;
          llvm::sort(commutative_ops, [](auto lhs, auto rhs) {
            return lhs->isBeforeInBlock(rhs);
          });
          // find matched commutative ops, we find all of matched input values
          // find all users of matched inputs in commutative_ops
          llvm::SmallVector<mlir::Operation*, 4> matched_ops;
          if (origin_operand_size != commutative_ops.size() + 1) {
            for (auto op : commutative_ops) {
              for (auto operand : op->getOperands()) {
                if (matched_operands.end() !=
                        std::find(matched_operands.begin(),
                                  matched_operands.end(), operand) ||
                    matched_ops.end() !=
                        llvm::find_if(matched_ops, [&](auto op) {
                          return op->getResult(0) == operand;
                        })) {
                  matched_ops.push_back(op);
                }
              }
            }
            // find after match remaind operand, this operand must match
            // commutative ops's input and is not matched ops
            for (auto op : matched_ops) {
              for (auto operand : op->getOperands()) {
                if (llvm::find(matched_ops, operand.getDefiningOp()) ==
                        matched_ops.end() &&
                    matched_operands.end() ==
                        std::find(matched_operands.begin(),
                                  matched_operands.end(), operand)) {
                  rhs_operands_backup.push_back(operand);
                }
              }
            }
          } else {
            matched_ops.insert(matched_ops.end(), commutative_ops.begin(),
                               commutative_ops.end());
            for (auto operand : rhs_operands) {
              if (matched_operands.end() ==
                      llvm::find(matched_operands,
                                 operand) &&  // avoid redundant add
                  rhs_operands_backup.end() ==
                      llvm::find(rhs_operands_backup, operand)) {
                rhs_operands_backup.push_back(operand);
              }
            }
          }
          insert_fused_ops(matched_ops.begin(), matched_ops.end());

          //  arg0     arg1 dot     add
          //   \        |    \       |
          //   mul     mul   mul    mul1
          //    \      /      \     /
          //       mul          mul
          // if it has over two args, we can't judg match(arg0, dot) or
          // match(arg0, add) is matched, We have to use other patterns to
          // match.
          if (lhs_iter != lhs_operands.end()) {
            size_t args_num = lhs_iter->template cast<mlir::BlockArgument>()
                                  .getOwner()
                                  ->getNumArguments();
            llvm::SmallVector<size_t, 64> bucked(args_num, 0);
            for (auto it : llvm::make_range(lhs_iter, lhs_operands.end())) {
              ++bucked[it.template cast<mlir::BlockArgument>().getArgNumber()];
            }
            if (1 < llvm::count_if(bucked, [](auto it) { return it != 0; })) {
              continue;
            }
          }

          lhs_operands.erase(lhs_operands.begin(), lhs_iter);
          rhs_operands.swap(rhs_operands_backup);
          assert(lhs_operands.size() == rhs_operands.size() &&
                 "remind arguments size mismatch!");
        }
        for (const auto& it : llvm::zip(lhs_operands, rhs_operands)) {
          queue.emplace_front(std::get<0>(it), std::get<1>(it));
        }
      } else {
        lhs_operands.assign(
            std::begin(cur.first.getDefiningOp()->getOperands()),
            std::end(cur.first.getDefiningOp()->getOperands()));
        rhs_operands.assign(
            std::begin(cur.second.getDefiningOp()->getOperands()),
            std::end(cur.second.getDefiningOp()->getOperands()));
        for (const auto& it : llvm::zip(lhs_operands, rhs_operands)) {
          queue.emplace_back(std::get<0>(it), std::get<1>(it));
        }
      }
    }
    return true;
  }
};
 