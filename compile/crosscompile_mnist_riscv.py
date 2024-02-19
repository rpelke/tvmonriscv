import tensorflow as tf
import tvm
import tvm.relay.testing.tf as tf_testing
import numpy as np

import os
repo_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'))


def build_lq_lib(nn_model, nn_name: str, batch: int, store_path: str, cc: str) -> str :
    """Build a library for a given nn model and batch size

    Args:
        lq_nn: NN model (TensorFlow)
        nn_name (str): Name of NN model
        batch (int): Input batch size
        store_path (str): Store path of .so
        cc (str): Path of cross compiler

    Returns:
        str: Filename of created lib
    """
    layout = 'NHWC'
    in_shapes = []

    for layer in nn_model._input_layers:
        if tf.executing_eagerly():
            in_shapes.append(tuple(dim if dim is not None else batch for dim in layer.input.shape))
        else:
            in_shapes.append(
                tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape)
            )

    def __build_lib(in_data, target, dev, dtype="float32"):
        shape_dict = {name: x.shape for (name, x) in zip(nn_model.input_names, in_data)}
        print(shape_dict)
        mod, params = tvm.relay.frontend.keras.from_keras(model=nn_model, shape=shape_dict, layout=layout)
        
        print(mod)
        
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.relay.build(mod, target=target, params=params)
        
        inp_shape_str = ''.join([str(i) + 'x' for i in in_data[0].shape])[:-1]
        lib_name = f'{nn_name}_input_{inp_shape_str}_lib.so'
        store_file = f'{store_path}/{lib_name}'
        lib.export_library(store_file, cc=cc)
        print(mod)
        return lib_name

    in_data = [np.random.uniform(size=shape, low=-1.0, high=1.0) for shape in in_shapes]

    # Cross-compile for RISC-V
    target = tvm.target.Target(
        "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"
    )
    lib_name = __build_lib(in_data, target, None)
    return lib_name


nn_name = 'mnist_cnn'
cc_path = os.environ['CXX']
model = tf.keras.models.load_model(f'{repo_path}/models/{nn_name}.h5')
lib_name = build_lq_lib(model, nn_name=nn_name, batch=1, store_path=f'{repo_path}/models', cc=cc_path)
print(f'Created lib {lib_name} in {repo_path}/models/')
