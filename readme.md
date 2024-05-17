## TVM Meets RISC-V - Getting Started
This repository contains a minimum example of how to build a CNN, compile it using TVM, and to run it on RISC-V (user mode QEMU) using the TVM runtime.
There is also an example how to generate C code with TVM to use the auto-vectorization features of clang if you want to have RISC-V vector instructions.


### Build LLVM and TVM
Since TVM uses files from the LLVM project, not all TVM and LLVM versions are compatible.
The versions can be changed in the `build_*`scripts.

```bash
chmod +x build_llvm.sh
chmod +x build_tvm.sh
./build_llvm.sh
./build_tvm.sh
python3 -m venv .venv
pip3 install -r requirements.txt
```
The following environment variables should be set:
```bash
export LD_LIBRARY_PATH=${PWD}/tvm/build/debug/lib
export TVM_LIBRARY_PATH=${PWD}/tvm/build/debug/build
export PYTHONPATH=${PWD}/tvm/python
```


### Train Model (Offline)
First, we need to train a neural network model. We use TensorFlow for this purpose. This should happen offline, before the simulation.
The output will be a .h5 file of the model. It is stored under `models/mnist_cnn.h5`.
```bash
python3 train/train_mnist.py
```


### Compile and Execute Model for Host Execution
```bash
python3 compile/compile_mnist_host.py
python3 execute/execute_mnist.py 
```
A shared object (.so) file should now appear in the `models` folder.
You can check the properties of the lib:
```bash
nm -gD models/<lib_name>.so
ldd models/<lib_name>.so
```
The `ldd` command is used to display the shared library dependencies of the shared library `<lib_name>.so`.
The `nm` command is used to display symbol information of the shared library.


### Execute TVM Output on RISC-V QEMU (64 Bit)
Build the RISC-V toolchain to get a cross-compiler and QEMU:
```bash
cd tools
git clone --depth=1 https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
git checkout 710a81b
mkdir -p riscv-gnu-toolchain/build/release/build
cd riscv-gnu-toolchain/build/release/build
../../../configure --prefix=${PWD}/.. --enable-llvm
make -j `nproc` linux  
make -j `nproc` build-sim SIM=qemu
cd ../../../../../
export CXX=${PWD}/tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-g++
export TVM_ROOT=${PWD}/tvm
export DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
```

Cross compile the NN for RISC-V and start using QEMU:
```bash
cd execute
make lib/libtvm_runtime_pack.o
make lib/cpp_mnist_pack
```
Further information about deployment using the TVM C++ runtime can be found [here](https://github.com/apache/tvm/tree/main/apps/howto_deploy).
Execute ML model:
```bash
python3 compile/crosscompile_mnist_riscv.py
cd models
../tools/riscv-gnu-toolchain/build/release/bin/qemu-riscv64 ../execute/lib/cpp_mnist_pack
```


### Use RISC-V Vector Extension
The current TVM version does not support the RISC-V vector extension. To use vector instructions, we use TVM to compile to C code. Then, we use a Clang cross-compiler to make use of LLVM's auto-vectorization features.

```bash
python3 compile/compile_to_c.py
```
You have to write a main function for the resulting c file that calls `__tvm_main__`. A dirty example done by me is shown in `execute/vec_add_c.c`. The input and output arrays are marked as `volatile`. Otherwise, the compiler could optimize away the vector instructions in this small example.

To run the example, execute the following steps:
```bash
export CXX_CLANG_RISCV=${PWD}/tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-clang++
cd execute
make lib/libtvm_runtime_pack_riscv.o
make lib/vec_add_c
../tools/riscv-gnu-toolchain/build/release/bin/qemu-riscv64 lib/vec_add_c
```

It may be necessary to make the vec_add_c file executable before simulation:
```bash
chmod +x lib/vec_add_c
../tools/riscv-gnu-toolchain/build/release/bin/qemu-riscv64 lib/vec_add_c
```

To disassemble files containing instructions from the vector extension, the following might work better than the `objdump` from the RISC-V toolchain:
```bash
llvm/build/release/bin/llvm-objdump -d models/mnist_cnn_input_1x28x28x1_lib.so
````
