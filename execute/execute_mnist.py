import tvm
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_executor
import numpy as np

import os
repo_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'))


# Step 1: Load MNIST Dataset (offline)
# Don't load TF in the simulator. Use random inputs instead.
import tensorflow as tf
nn_name = 'mnist_cnn'
input_tensor = tf.keras.layers.Input(shape=(28, 28, 1))
model = tf.keras.models.load_model(f'{repo_path}/models/{nn_name}.h5')
model.summary()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images / 127.5 - 1

# Test input image consisting of only 0.5s
batch = 1
input_05 = 0.5 * np.ones((batch,28,28,1))
out_05 = model.predict(input_05)
print(out_05)

# Step 2: Load lib (simulator)
in_data = [train_images[0:batch, :, :, :]]
inp_shape_str = ''.join([str(i) + 'x' for i in in_data[0].shape])[:-1]
lib_name = f'{nn_name}_input_{inp_shape_str}_lib.so'

target, dev = tvm.testing.enabled_targets()[0]
lib: tvm.runtime.Module = tvm.runtime.load_module(f"{repo_path}/models/{lib_name}")
m = graph_executor.GraphModule(lib["default"](dev))
for name, x in zip(model.input_names, in_data):
    m.set_input(name, tvm.nd.array(x.astype("float32")))
m.run()
tvm_out = [m.get_output(i).numpy() for i in range(m.get_num_outputs())]
tvm_out = tvm_out[0]


# Step 3: Test outputs against TF (offline)
tf_out = model.predict(train_images[0:batch, :, :, :])

for kout, tout in zip(tf_out, tvm_out):
    tvm.testing.assert_allclose(kout, tout, rtol=1e-5, atol=1e-5)
print("TVM test successful.")
