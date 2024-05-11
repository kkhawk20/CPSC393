import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Flatten, Dense, Reshape
import keras_tuner as kt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['TF_GRAPPLER_OPTIMIZERS'] = 'constfold,debug_stripper,remap,inlining,memory_optimization,arithmetic,autoparallel,dependency,loop_optimizer'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.enable_eager_execution()

print("Eager execution: ", tf.executing_eagerly())
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

# Clear the TensorFlow session
tf.keras.backend.clear_session()

main_path = "/app/rundir/CPSC393/FinalProject/"
batch_size = 2
num_classes = 24
latent_dim = 100

train_df = pd.read_csv("./sign_mnist_train.csv")
test_df = pd.read_csv("./sign_mnist_test.csv")

# Separating X and Y
y_train = train_df.pop('label')
y_test = test_df.pop('label')

# Rescale data to be 0-1 instead of 0-255
trainX = train_df.values.reshape(-1, 28, 28, 1) / 255.0
testX = test_df.values.reshape(-1, 28, 28, 1) / 255.0

# Change the labels to be in the correct format
lb = LabelBinarizer()
trainY = lb.fit_transform(y_train)
testY = lb.transform(y_test)

# Visualize some images
f, ax = plt.subplots(2,5)
f.set_size_inches(10, 10)
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(trainX[j + i * 5].reshape(28, 28), cmap="gray")
    plt.tight_layout()    
plt.savefig("ASL_MNIST_Images.png")

# Use tf.data to create the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

def create_base_model(input_shape=(28, 28, 1), num_classes=24):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Prepare the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY)).shuffle(1024).batch(32)

# Create and train the base model
base_model = create_base_model()
base_model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
base_model.fit(train_dataset, epochs=10)

# Save the weights
base_model.save_weights('base_model_weights.h5')

def build_generator(hp, base_model):
    # Create a new model that reuses the convolutional layers of the base model
    conv_model = keras.Sequential(base_model.layers[:-2])  # Exclude the last two layers
    for layer in conv_model.layers:
        layer.trainable = False  # Freeze the layers

    model = keras.Sequential([
        keras.Input(shape=(latent_dim,)),
        Dense(7*7*64, activation='relu'),
        Reshape((7, 7, 64)),
        Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid'),
        conv_model  # Append the pretrained convolutional model
    ])
    return model

def build_discriminator():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = discriminator_loss
        self.g_loss_fn = generator_loss

    def train_step(self, data):
        images, labels = data
        batch_size = tf.shape(images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            d_loss = self.d_loss_fn(real_output, fake_output)
            g_loss = self.g_loss_fn(fake_output)

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Initialize and compile the GAN
discriminator = build_discriminator()
# Load the base model
base_model = create_base_model()
base_model.load_weights('base_model_weights.h5')

# Build the generator with transfer learning
generator = build_generator(kt.HyperParameters(), base_model)

# Initialize the GAN
gan = ConditionalGAN(discriminator=build_discriminator(), generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(0.0001),
    g_optimizer=keras.optimizers.Adam(0.0001)
)

# Train the GAN
gan.fit(train_dataset, epochs=50)

def generate_and_plot_images(generator, word, label_map, latent_dim, grid_dim=(2, 5), file_name="generated_images.png"):
    # Split the word into letters and convert each letter to its corresponding label
    labels = [label_map[letter] for letter in word.lower() if letter in label_map]

    # Create random noise
    noise = tf.random.normal([len(labels), latent_dim])

    # Generate images
    images = generator.predict(noise, batch_size=len(labels))
    images = (images * 255).astype(np.uint8)

    # Create a grid of images
    fig, axes = plt.subplots(grid_dim[0], grid_dim[1], figsize=(15, 10))
    for img, ax, letter in zip(images, axes.flatten(), word):
        ax.imshow(img[:, :, 0], cmap='gray')  # Adjust indexing if necessary
        ax.set_title(f"Sign for '{letter}'")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(file_name)

# Example label dictionary (you need to adjust this according to your dataset)
label_dict = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'w': 22,
    'x': 23
}
# # Load the generator model
# generator = gan.generator()

# Generate and plot images for a word
generate_and_plot_images(generator, "hello", label_dict, latent_dim=latent_dim, grid_dim=(1, 5), file_name="generated_images_hello.png")
