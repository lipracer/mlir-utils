include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"fusion", "ModuleOp"> {
  let summary = "fusion pass";
  let constructor = "createFusionPass()";
}

def AutoFusionPass : Pass<"auto-fusion", "ModuleOp"> {
  let summary = "auto fusion pass";
  let constructor = "createAutoFusionPass()";
}

def DenseCSEPass : Pass<"dense-cse", "FuncOp"> {
  let summary = "cse for dense graph.";
  let constructor = "createDenseCSEPass()";
}