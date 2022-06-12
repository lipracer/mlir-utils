include "mlir/Pass/PassBase.td"

def FusionPass : Pass<"fusion", "ModuleOp"> {
  let summary = "fusion pass";
  let constructor = "createFusionPass()";
}