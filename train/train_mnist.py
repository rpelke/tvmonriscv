import tensorflow as tf
from pathlib import Path
import numpy as np

import os
repo_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'))


# Load MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

# Training parameters
batch_size = 64
num_classes = 10
epochs = 3
dtype = np.float32

# Define model structure
inputs = tf.keras.Input(shape=(28, 28, 1), dtype=dtype)
model = tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', dtype=dtype)(inputs)
model = tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', dtype=dtype)(model)
model = tf.keras.layers.MaxPool2D()(model)
model = tf.keras.layers.Dropout(0.25)(model)
model = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', dtype=dtype)(model)
model = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', dtype=dtype)(model)
model = tf.keras.layers.MaxPool2D(strides=(2,2))(model)
model = tf.keras.layers.Dropout(0.25)(model)
model = tf.keras.layers.Flatten()(model)
model = tf.keras.layers.Dense(128, activation='relu', dtype=dtype)(model)
model = tf.keras.layers.Dropout(0.5)(model)
model = tf.keras.layers.Dense(num_classes, activation='softmax', dtype=dtype)(model)
    
# Show model structure
model = tf.keras.Model(inputs=inputs, outputs=model)
tf.keras.Model.summary(model)

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels))

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f} %")

# Save model
nn_name = 'mnist_cnn'
Path(f'{repo_path}/models').mkdir(parents=True, exist_ok=True)
model.save(f'{repo_path}/models/{nn_name}.h5')
