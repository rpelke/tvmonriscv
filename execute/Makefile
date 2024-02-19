PKG_CFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread

.PHONY: clean all

all: lib/cpp_deploy_pack lib/cpp_deploy_normal

# Build rule for all in one TVM package library
lib/libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

# Deploy using the all in one TVM package library
lib/cpp_mnist_pack: execute_mnist.cc lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# Deploy using pre-built libtvm_runtime.so
lib/cpp_mnist_normal: execute_mnist.cc lib/test_addone_sys.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)
clean:
	rm -rf lib


# Build C Code
lib/test_so: test.c
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime -L${TVM_ROOT}/build/debug/lib

lib/test: test.c lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# Builf for RISC-V
CLANG_FLAGS = -march=rv64imafcv
lib/libtvm_runtime_pack_riscv.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX_CLANG_RISCV) $(CLANG_FLAGS) -c $(PKG_CFLAGS) -o $@  $^

lib/vec_add_c: vec_add_c.c lib/libtvm_runtime_pack_riscv.o
	@mkdir -p $(@D)
	$(CXX_CLANG_RISCV) $(CLANG_FLAGS) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)
