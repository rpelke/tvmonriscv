cmake_minimum_required(VERSION 3.10)

project(TVMStaticBuild)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED TVM_ROOT)
    message(FATAL_ERROR "TVM_ROOT must be set. Use -DTVM_ROOT=/path/to/tvm.")
endif()

include_directories(
    ${TVM_ROOT}
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${TVM_ROOT}/3rdparty/dlpack/include
)

set(SOURCES_RT
    execute/tvm_runtime_pack.cc
)

# Create packed exeutable
link_directories(${TVM_ROOT}/build)
add_library(tvm_runtime_pack SHARED ${SOURCES_RT})
target_compile_definitions(tvm_runtime_pack PUBLIC -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
install(TARGETS tvm_runtime_pack DESTINATION lib)

# Create packed exeutable
add_executable(nn_packed execute/execute_mnist.cc)
target_link_libraries(nn_packed PRIVATE tvm_runtime_pack dl pthread)
install(TARGETS nn_packed DESTINATION bin)
