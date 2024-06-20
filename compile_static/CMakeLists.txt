project(compile-static)
cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_FLAGS_DEBUG -O2)

if(NOT CRT_ROOT)
    message(FATAL_ERROR "CRT_ROOT is not set. Set it to the <tvm_build>/standalone_crt folder.")
endif()


set(CRT_CONFIG_PATH ${CMAKE_CURRENT_BINARY_DIR}/crt_include)
file(MAKE_DIRECTORY ${CRT_CONFIG_PATH})
configure_file(
    ${CRT_ROOT}/../microtvm_template_projects/crt/crt_config/crt_config.h
    ${CRT_CONFIG_PATH}
    COPYONLY
)

if(NOT DEFINED ENV{TVM_LIBRARY_PATH})
    message(FATAL_ERROR "TVM_LIBRARY_PATH is not set.")
endif()

set(MOD_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/mod_libs/devc.o
    ${CMAKE_CURRENT_BINARY_DIR}/mod_libs/lib0.o
    ${CMAKE_CURRENT_BINARY_DIR}/mod_libs/lib1.o
)

add_custom_command(OUTPUT ${MOD_LIBS} build_model_output/graph_c.json build_model_output/params_c.bin
    COMMAND PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tvm/python
            python3 "${CMAKE_CURRENT_SOURCE_DIR}/build_model.py" -o build_model_output &&
            mkdir mod_libs && cd mod_libs && tar -xf ../build_model_output/model_c.tar
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build_model.py"
)

add_custom_command(OUTPUT build_model_output/graph_c.json.c build_model_output/params_c.bin.c
    COMMAND xxd -i graph_c.json > graph_c.json.c
    COMMAND xxd -i params_c.bin > params_c.bin.c
    WORKING_DIRECTORY build_model_output
    DEPENDS build_model_output/graph_c.json build_model_output/params_c.bin
)

set(CRT_LIBS
    memory
    graph_executor
    common
)
add_subdirectory(${CRT_ROOT} standalone_crt/)

add_executable(demo_static
    ${CMAKE_CURRENT_SOURCE_DIR}/demo_static.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bundle_static.c
    ${CMAKE_CURRENT_BINARY_DIR}/build_model_output/graph_c.json.c
    ${CMAKE_CURRENT_BINARY_DIR}/build_model_output/params_c.bin.c
)
target_compile_definitions(demo_static PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

target_link_libraries(demo_static ${MOD_LIBS} ${CRT_LIBS} m pthread)

if(NOT DEFINED ENV{TVM_ROOT})
    message(FATAL_ERROR "TVM_ROOT is not set.")
endif()
set(TVM_ROOT $ENV{TVM_ROOT})

if(NOT DEFINED ENV{DMLC_CORE})
    message(FATAL_ERROR "DMLC_CORE is not set.")
endif()
set(DMLC_CORE $ENV{DMLC_CORE})

target_include_directories(demo_static PUBLIC
    ${TVM_ROOT}/include
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${DMLC_CORE}/include
    ${CRT_CONFIG_PATH}
)
