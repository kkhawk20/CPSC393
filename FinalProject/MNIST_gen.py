import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2DTranspose, TimeDistributed, Conv2D, Flatten, Dense, LSTM
import keras_tuner as kt
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
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
num_channels = 1
num_classes = 408
image_size = 256
latent_dim = 408
max_frames = 16

train_df = pd.read_csv("./sign_mnist_train.csv")
test_df = pd.read_csv("./sign_mnist_test.csv")

# Separating X and Y
y_train = train_df['label']
y_test = test_df['label']

del train_df['label']
del test_df['label']

# rescale data to be 0-1 instead of 0-255
trainX = train_df.astype("float32") / 255.0
testX = test_df.astype("float32") / 255.0

# change the labels to be in the correct format
lb = LabelBinarizer()
trainY = lb.fit_transform(y_train)
testY = lb.transform(y_test)

trainX.head()
trainX.shape

print(trainX.shape,
trainY.shape)

print(testX.shape,
testY.shape)

# Visualize some images!!!
import matplotlib.pyplot as plt

# I used different names cuz i wanted to reshape them without
# Changing the original data put into the model :)
x_train = train_df.values
x_test = test_df.values
x_train_vis = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
f, ax = plt.subplots(2,5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train_vis[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()    
plt.savefig("ASL_MNIST_Images.png")

# Convert the pandas DataFrame to numpy arrays
x_train = trainX.values.reshape(-1, 28, 28, 1)
y_train = trainY

# Use tf.data to create the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Batch and shuffle the dataset
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

def build_generator(hp):
    latent_dim = 408
    num_classes = 408
    seq_length = 16
    height = 256
    width = 256
    channels = 3

    noise_input = keras.Input(shape=(latent_dim,))
    label_input = keras.Input(shape=(num_classes,))
    concat_input = layers.Concatenate()([noise_input, label_input])

    # Dense layer to expand the input
    x = layers.Dense(4 * 4 * 256, activation='relu')(concat_input)
    x = layers.Reshape((1, 4, 4, 256))(x)

    # ConvLSTM layers for sequence generation
    x = layers.ConvLSTM2D(128, (3, 3), padding="same", return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    print(f"Shape after ConvLSTM layers: {x.shape}")

    # Upscaling to the desired dimensions
    x = TimeDistributed(Conv2DTranspose(64, (5, 5), strides=(4, 4), padding='same', activation='relu'))(x)
    print(f"After first Conv2DTranspose: {x.shape}")
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu'))(x)
    print(f"After second Conv2DTranspose: {x.shape}")
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', activation='relu'))(x)
    print(f"After third Conv2DTranspose: {x.shape}")
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', activation='relu'))(x)
    print(f"After fourth Conv2DTranspose: {x.shape}")
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', activation='sigmoid'))(x)
    print(f"Final generator output shape: {x.shape}")

    # Repeat the generated frames to match the expected sequence length
    x = layers.Lambda(lambda x: tf.repeat(x, seq_length, axis=1))(x)

    model = keras.Model(inputs=[noise_input, label_input], outputs=x)
    return model

def build_discriminator(seq_length=16, height=256, width=256, channels=3):
    model = keras.Sequential([
        keras.Input(shape=(seq_length, height, width, channels)),
        TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")),
        TimeDistributed(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")),
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=False),
        Flatten(),
        Dense(1)
    ])
    return model

def numpy_safe(tensor):
    return tf.py_function(lambda x: x.numpy(), [tensor], Tout=[tf.float32])

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.d_loss_history = []
        self.g_loss_history = []

    def call(self, inputs, training=False):
        # inputs should be a list with two elements: [noise_input, label_input]
        if isinstance(inputs, list) and len(inputs) == 2:
            noise_input = inputs[0]
            label_input = inputs[1]
        else:
            raise ValueError("Expected inputs to be a list of [noise_input, label_input]")

        generated_videos = self.generator([noise_input, label_input], training=training)

        if training:
            real_inputs = inputs[0]  # Assuming real inputs (videos) are passed as the first part of the input during training
            real_outputs = self.discriminator(real_inputs, training=True)
            fake_outputs = self.discriminator(generated_videos, training=True)
            return real_outputs, fake_outputs
        else:
            return generated_videos


    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        real_videos, _ = data
        batch_size = tf.shape(real_videos)[0]
        random_labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
        label_one_hot = tf.one_hot(random_labels, depth=num_classes)
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_videos = self.generator([random_latent_vectors, label_one_hot], training=True)
            real_predictions = self.discriminator(real_videos, training=True)
            fake_predictions = self.discriminator(generated_videos, training=True)
            d_loss_real = self.d_loss_fn(tf.ones_like(real_predictions), real_predictions)
            d_loss_fake = self.d_loss_fn(tf.zeros_like(fake_predictions), fake_predictions)
            d_loss = d_loss_real + d_loss_fake
            g_loss = self.g_loss_fn(fake_predictions)

        # Gradient application
        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        # Update the metrics
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        # Save history for plotting
        d_loss = self.d_loss_tracker.result()
        d_loss_np = numpy_safe(d_loss)[0]
        self.d_loss_history.append(d_loss_np)

        g_loss = self.g_loss_tracker.result()
        g_loss_np = numpy_safe(g_loss)[0]
        self.g_loss_history.append(g_loss_np)

        # Reset the metrics at the end of each batch
        self.d_loss_tracker.reset_states()
        self.g_loss_tracker.reset_states()

        return {"d_loss": d_loss, "g_loss": g_loss}

def discriminator_loss(real_output, fake_output):
    smooth_positive_labels = 0.9
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output) * smooth_positive_labels, real_output, from_logits=True)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

def build_gan(hp):
    generator = build_generator(hp)
    discriminator = build_discriminator()

    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    batch_size = hp.Choice("batch_size", [2, 4, 8])

    gan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        g_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss
    )

    return gan

tuner = kt.Hyperband(
    build_gan,
    objective=kt.Objective("g_loss", direction="min"),
    max_epochs=30,
    hyperband_iterations=2,
    overwrite=True
)

retrain = True

if retrain:

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='g_loss', patience=20)
    tuner.search(train_dataset, epochs=20, callbacks=[stop_early])
    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.generator.save_weights("generator_weights_tuned.h5")
    best_model.generator.save("generator_model_tuned.h5")
    best_model.discriminator.save_weights("discriminator_weights_tuned.h5")

    # Also saving a plot showing the losses over epochs
    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(best_model.d_loss_history, label='Discriminator Loss')
    plt.plot(best_model.g_loss_history, label='Generator Loss')
    plt.title('Training Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses_over_epochs.png')

else :
    print("Lets do some predictions!!")

    def generate_and_plot_images(generator, word, latent_dim=100, num_images=10, grid_dim=(2, 5), fileName = "generated_images.png"):

        # Split the given word into letters
        word_list = list(word.lower())

        # Create random noise and one-hot labels
        noise = tf.random.normal([num_images, latent_dim])
        label_one_hot = tf.one_hot([word_label] * num_images, depth=num_classes)

        # Validate shapes
        print(f"Noise shape: {tf.shape(noise)}, Label shape: {tf.shape(label_one_hot)}")

        # Generate images
        inputs = [noise, label_one_hot]
        images = generator.predict(inputs, batch_size=num_images)
        images = (images * 255).astype(np.uint8)

        images = images[0]  # Assuming only one batch and selecting the first video's frames

        # Create a grid of images
        fig, axes = plt.subplots(grid_dim[0], grid_dim[1], figsize=(15, 10))
        for img, ax in zip(images, axes.flatten()):
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(fileName)

    # Load generator model
    generator = load_model("generator_weights_tuned.h5", compile=False)

word_list = ['tired']
for word in word_list:
    fileName = f"generated_images_tuned_{word}.png"
    generate_and_plot_images(generator, word, label_dict, latent_dim=latent_dim, num_images=10, grid_dim=(2, 5), fileName=fileName)

