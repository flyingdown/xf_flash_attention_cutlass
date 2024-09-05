# cmake .. -DCMAKE_CXX_COMPILER=${ROCM_PATH}/bin/hipcc -DAMDGPU_TARGETS=gfx928 -DCMAKE_INSTALL_PREFIX=pa_install -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(paged-attention LANGUAGES CXX)

find_package(HIP REQUIRED)

# 设置 CMake 的 C/C++ 编译器变量为 hipcc
set(CMAKE_C_COMPILER hipcc)
set(CMAKE_CXX_COMPILER hipcc)

###################### test ####################
# 需要找到torch threads pybind 
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
# 明确指定 Python 库路径
find_library(PYTHON_LIB python3.10 PATHS /usr/local/lib)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

file(
  GLOB_RECURSE
  PG_SOURCES_HIP
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src/*.cpp)

add_library(paged-attention SHARED
    export.cpp
    csrc/paged_attn.cpp
    ${PG_SOURCES_HIP}
  )

target_include_directories(paged-attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_3.2.1/include/
    /usr/local/lib/python3.10/site-packages/torch/include/
    /usr/local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include
    /usr/local/include/python3.10/
    /usr/local/lib/python3.10/site-packages/pybind11/include
    ${HIP_INCLUDE_DIRS})

target_link_libraries(paged-attention PRIVATE hip::device)

###################### test ####################
# 链接libtorch库 pthread库
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    Threads::Threads
    ${PYTHON_LIB} 
  )

target_compile_options(paged-attention PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -Wno-deprecated-builtins
  -Wno-deprecated
  -DNDEBUG
  -DUSE_ROCM
  -U__HIP_NO_HALF_OPERATORS__
  -U__HIP_NO_HALF_CONVERSIONS__
  -fPIC
  -O3
  -std=c++17
  -D__HIP_PLATFORM_HCC__=1
  --offload-arch=gfx928
  -DDCU_ASM
  -DHIP_ENABLE_WARP_SYNC_BUILTINS
  -DFLASHATTENTION_DISABLE_BACKWARD
  -ffast-math
  #-mllvm 
  #-enable-num-vgprs-512=true
  >)

INSTALL(TARGETS paged-attention
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

INSTALL(FILES csrc/paged_attn.h DESTINATION "${CMAKE_INSTALL_PREFIX}/include")
