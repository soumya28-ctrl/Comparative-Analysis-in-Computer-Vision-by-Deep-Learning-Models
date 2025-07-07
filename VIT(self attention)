In a Vision Transformer, self-attention allows the model to weigh the importance of different parts of the input
image (represented as a sequence of flattened patches) when processing each individual part. Instead of relying on fixed, local receptive fields like convolutional neural networks,
self-attention enables each patch to attend to all other patches in the image. This is achieved by calculating three vectors for each patch: a query (Q), a key (K), and a value (V). 
The similarity between a patch's query vector and all other patches' key vectors determines the attention weight. These weights are then used to compute a weighted sum of the value vectors
, effectively creating a new representation for the patch that incorporates information from the entire image, capturing long-range dependencies and global context.


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow as tf

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Patch parameters
patch_size = 7  # 28x28 -> 4x4 patches if patch size is 7
num_patches = (28 // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [128, 64]
transformer_layers = 4

# Helper: Create image patches
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Helper: Patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# Build the ViT model
inputs = layers.Input(shape=(28, 28, 1))
patches = Patches(patch_size)(inputs)
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

# Transformer blocks
for _ in range(transformer_layers):
    # Layer normalization 1
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
    # Skip connection 1
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP
    x3 = layers.Dense(transformer_units[0], activation='relu')(x3)
    x3 = layers.Dense(transformer_units[1], activation='relu')(x3)
    # Skip connection 2
    encoded_patches = layers.Add()([x3, x2])

# Classification head
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = layers.Flatten()(representation)
representation = layers.Dense(128, activation='relu')(representation)
logits = layers.Dense(10, activation='softmax')(representation)

model = models.Model(inputs=inputs, outputs=logits)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)

# Predict and display
predictions = model.predict(x_test)
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import tensorflow as tf

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Patch parameters
patch_size = 7  # 28x28 -> 4x4 patches if patch size is 7
num_patches = (28 // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [128, 64]
transformer_layers = 4

# Helper: Create image patches
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Helper: Patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# Build the ViT model
inputs = layers.Input(shape=(28, 28, 1))
patches = Patches(patch_size)(inputs)
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

# Transformer blocks
for _ in range(transformer_layers):
    # Layer normalization 1
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
    # Skip connection 1
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP
    x3 = layers.Dense(transformer_units[0], activation='relu')(x3)
    x3 = layers.Dense(transformer_units[1], activation='relu')(x3)
    # Skip connection 2
    encoded_patches = layers.Add()([x3, x2])

# Classification head
representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
representation = layers.Flatten()(representation)
representation = layers.Dense(128, activation='relu')(representation)
logits = layers.Dense(10, activation='softmax')(representation)

model = models.Model(inputs=inputs, outputs=logits)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)

# Predict and display
predictions = model.predict(x_test)
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
OUTPUT:
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Epoch 1/5
844/844 ━━━━━━━━━━━━━━━━━━━━ 32s 17ms/step - accuracy: 0.8661 - loss: 0.4224 - val_accuracy: 0.9743 - val_loss: 0.0909
Epoch 2/5
844/844 ━━━━━━━━━━━━━━━━━━━━ 7s 8ms/step - accuracy: 0.9736 - loss: 0.0807 - val_accuracy: 0.9822 - val_loss: 0.0653
Epoch 3/5
844/844 ━━━━━━━━━━━━━━━━━━━━ 10s 8ms/step - accuracy: 0.9835 - loss: 0.0520 - val_accuracy: 0.9830 - val_loss: 0.0641
Epoch 4/5
844/844 ━━━━━━━━━━━━━━━━━━━━ 6s 7ms/step - accuracy: 0.9860 - loss: 0.0434 - val_accuracy: 0.9835 - val_loss: 0.0570
Epoch 5/5
844/844 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.9870 - loss: 0.0377 - val_accuracy: 0.9843 - val_loss: 0.0535
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step
