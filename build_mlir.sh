LLVM_BUILD_PATH="llvm-build"

if [ ! -d "$LLVM_BUILD_PATH" ]; then
mkdir $LLVM_BUILD_PATH
fi

cmake -GNinja -S ./third_party/llvm-project/llvm \
    -B llvm-build \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DLLVM_INCLUDE_TOOLS=ON \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=gcc-7 \
    -DCMAKE_CXX_COMPILER=g++-7

cmake --build llvm-build --target all -- -j8