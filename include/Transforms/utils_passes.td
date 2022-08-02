include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"fusion", "ModuleOp"> {
  let summary = "fusion pass";
  let constructor = "createFusionPass()";
}

def AutoFusionPass : Pass<"auto-fusion", "ModuleOp"> {
  let summary = "auto fusion pass";
  let constructor = "createAutoFusionPass()";
}