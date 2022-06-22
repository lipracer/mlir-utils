module {
  func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1152x1024xf32>) -> tensor<1152x1024xf32> {
    %0 = arith.constant dense<1.11111116> : tensor<1xf32>
    %2 = arith.constant dense<[1152, 1024]> : tensor<2xi32>
    %4 = arith.constant dense<1.000000e-01> : tensor<1xf32>
    %6 = "tosa.greater_equal"(%arg2, %4) : (tensor<1152x1024xf32>, tensor<1xf32>) -> tensor<1152x1024xi1>
    %7 = "tosa.cast"(%6) : (tensor<1152x1024xi1>) -> tensor<1152x1024xf32>
    %8 = "tosa.mul"(%0, %7) { shift = 1 : i32 } : (tensor<1xf32>, tensor<1152x1024xf32>) -> tensor<1152x1024xf32>
    return %8 : tensor<1152x1024xf32>
  }
}