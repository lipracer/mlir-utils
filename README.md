cmake -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7 -S .. -B .

### TVM Auto Fusion

- create indexed forward graph  
` 
auto graph = IndexedForwardGraph::Create(&arena_, body);
`

  - topologic visit
    ```
    def visit(op):
        for arg in op.args:
            if visited(arg):
                continue
        visit(arg)
    visit(root)
    ```
  - update node and it's outputs and edge pattern  
    ```
    void Update(const Expr& node, IndexedForwardGraph::Node* parent, OpPatternKind pattern);
    ```
  - push current node into postorder queue.
    ```
    void AddNode(const tvm::Object* key)
    ```

- GraphPartitioner::Partition

  - init group  
    union find group
  - build dominator tree
    ```
    static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
        if (lhs > rhs) return lhs;
            return rhs;
    }
    for node in reverse(postorder):
        node.parent = LeastCommonAncestor(node.outputs)
    ```
    build dominatorTree Node:
    ```
    Node* GetNode(support::Arena* arena, IndexedForwardGraph::Node* gnode)

    // @param input_nodes current node's outputs
    // @param edge_pattern dominator node's OpPattern
    Node* LeastCommonAncestor(const LinkedList<IndexedForwardGraph::Edge>& input_nodes, OpPatternKind* edge_pattern)
    ```
  - RunFusion
    - phase 0
    ```
    // group_node is current visiting node in postorder queue.
    // dom_node OpPattern is a pttern of all of path which between current node and it's idom node.
    // group_node->pattern == kOutEWiseFusable && dom_node->pattern == kElemWise
    ```
    - phase 1
    - phase 2
    ```
    // Fuse injective ops into intermediate tuples, if any
    ```