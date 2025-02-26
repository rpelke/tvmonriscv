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

option(GPROF_CUSTOM_FUNCTIONS "Build with gprof and custom functions." OFF)

if(GPROF_CUSTOM_FUNCTIONS)
    # Compile functions.cc as shared library
    set(PREV_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    add_library(custom_functions SHARED compile/gprov/functions.cc)
    install(TARGETS custom_functions DESTINATION lib)

    # Create packed exeutable
    set(CMAKE_CXX_FLAGS "${PREV_CXX_FLAGS}")
    add_executable(nn_packed execute/execute_mnist.cc)
    target_link_libraries(nn_packed PRIVATE custom_functions tvm_runtime_pack dl pthread)
    install(TARGETS nn_packed DESTINATION bin)

else()
    # Create packed exeutable
    add_executable(nn_packed execute/execute_mnist.cc)
    target_link_libraries(nn_packed PRIVATE tvm_runtime_pack dl pthread)
    install(TARGETS nn_packed DESTINATION bin)
endif()
