
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_mlir_library(UtilsTransforms
  Fusion.cpp
  AutoFusion.cpp

DEPENDS
  MLIRUTILSPassIncGen
)