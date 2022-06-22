
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        UtilsTransforms
        utils
        )
add_llvm_executable(utils-opt utils-opt.cpp)
llvm_update_compile_flags(utils-opt)
target_link_libraries(utils-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(utils-opt)
