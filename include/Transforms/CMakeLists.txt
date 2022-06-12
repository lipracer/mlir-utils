set(LLVM_TARGET_DEFINITIONS utils_passes.td)
mlir_tablegen(utils_passes.h.inc -gen-pass-decls -name UTILS)
add_public_tablegen_target(MLIRUTILSPassIncGen)