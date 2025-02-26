#!/bin/bash

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
DIR=$DIR/../

rm -rf ${DIR}/build
rm ${DIR}/models/*.so
mkdir -p build/debug/build
cd build/debug/build

export TVM_ROOT=$DIR/tvm
export CXX=g++
export TVM_NUM_THREADS=1

# Set dbg flag -g and gprof flag -pg
cmake -DTVM_ROOT=$TVM_ROOT \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DGPROF_CUSTOM_FUNCTIONS=ON \
    -DCMAKE_CXX_FLAGS="-g -pg" \
    ../../../

make install

# Compile neural network
cd $DIR
export PYTHONPATH=${DIR}/tvm/python
export TVM_LIBRARY_PATH=${DIR}/tvm/build/debug/build
python3 compile/compile_mnist_host_gprof.py

# Native execution with gprof
export LD_LIBRARY_PATH=${DIR}/models:${DIR}/build/debug/lib
./build/debug/bin/nn_packed
gprof ./build/debug/bin/nn_packed gmon.out > analysis.txt
