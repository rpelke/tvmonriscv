#!/bin/env bash
set -e
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
        DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# Relase Build
mkdir -p $DIR/build/release/build
pushd $DIR/build/release/build > /dev/null

cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=${PWD}/../../../../tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${PWD}/../../../../tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-g++ \
    -DCMAKE_LINKER=${PWD}/../../../../tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-ld \
    -DCRT_ROOT=${PWD}/../../../../tvm/build/debug/build/standalone_crt \
    -DCMAKE_INSTALL_PREFIX=../ \
    ../../../

make -j `nproc` demo_static
popd > /dev/null

# Debug Build
mkdir -p $DIR/build/debug/build
cd $DIR/build/debug/build 
pushd $DIR/build/debug/build > /dev/null

cmake \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=${PWD}/../../../../tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${PWD}/../../../../tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-g++ \
    -DCMAKE_LINKER=${PWD}/../../../../tools/riscv-gnu-toolchain/build/release/bin/riscv64-unknown-linux-gnu-ld \
    -DCRT_ROOT=${PWD}/../../../../tvm/build/debug/build/standalone_crt \
    -DCMAKE_INSTALL_PREFIX=../ \
    ../../../

make -j `nproc` demo_static
popd > /dev/null
