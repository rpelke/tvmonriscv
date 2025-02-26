#!/bin/bash

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

rm -rf ${DIR}/build
rm ${DIR}/models/*.so
mkdir -p build/release/build
cd build/release/build

export TVM_ROOT=$DIR/tvm
export CXX=g++

cmake -DTVM_ROOT=$TVM_ROOT \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DCMAKE_CXX_COMPILER=${CXX} \
    ../../../

make install

# Compile neural network
cd $DIR
export PYTHONPATH=${DIR}/tvm/python
export TVM_LIBRARY_PATH=${DIR}/tvm/build/debug/build
python3 compile/compile_mnist_host.py

# Native execution
export LD_LIBRARY_PATH=${DIR}/models:${DIR}/build/release/lib
./build/release/bin/nn_packed
