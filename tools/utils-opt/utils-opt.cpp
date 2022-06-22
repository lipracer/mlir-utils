
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "include/passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::utils::registerAllUTILSPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  return failed(mlir::MlirOptMain(argc, argv, "MLIR pass driver\n", registry));
}
