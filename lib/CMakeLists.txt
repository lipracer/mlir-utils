
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_mlir_library(utils
  graph_matcher.cc
  diff_graph.cc)

add_subdirectory(Transforms)
