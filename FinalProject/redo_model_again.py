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


# Load the JSON file
with open(main_path + 'WLASL_v0.3.json', 'r') as data_file:
    json_data = json.load(data_file)

bbox_data = []
for entry in json_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        bbox = instance['bbox'] if 'bbox' in instance else None
        bbox_data.append({
            'video_id': video_id,
            'gloss': gloss,
            'bbox': bbox,
        })

bbox_df = pd.DataFrame(bbox_data)
bbox_df.set_index('video_id', inplace=True)

# Load label dictionary
label_dict_path = './labels_dict.txt'
label_dict = {}
with open(label_dict_path, 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        label_dict[key] = int(value)
inverse_label_dict = {v: k for k, v in label_dict.items()}

def load_video_data(directory, label_dict):
    video_files = []
    labels = []
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path):
            continue
        for video_id in os.listdir(category_path):
            video_path = os.path.join(category_path, video_id)
            if not os.path.isdir(video_path):
                continue
            frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
            if frame_files:
                video_files.append(frame_files)
                labels.append(label_dict.get(category, -1))  # Default to -1 if category not in dictionary
    return video_files, labels


class VideoDataGenerator(Sequence):
    def __init__(self, video_files, labels, batch_size, max_frames=16, n_channels=3, n_classes=408, shuffle=True):
        self.dim = (256, 256)
        self.video_files = video_files
        self.labels = labels
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_video_files_temp = [self.video_files[k] for k in indexes]
        X, y = self.__data_generation(list_video_files_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_video_files_temp):
        X = np.empty((self.batch_size, self.max_frames, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, video_files in enumerate(list_video_files_temp):
            video_frames = [self.load_and_process_frame(f) for f in video_files]
            video_frames = [f for f in video_frames if f is not None]

            if len(video_frames) < self.max_frames:
                padding_length = self.max_frames - len(video_frames)
                padding = np.zeros((padding_length, *self.dim, self.n_channels))
                video_frames = np.vstack((video_frames, padding))
            elif len(video_frames) > self.max_frames:
                video_frames = video_frames[:self.max_frames]

            X[i,] = video_frames
            y[i] = self.labels[i]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def load_and_process_frame(self, frame_file):
        try:
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.dim)
            frame = frame / 255.0
            return frame
        except Exception as e:
            print(f"Error processing {frame_file}: {e}")
            return None


main_path = "/app/rundir/CPSC393/FinalProject/images/"
video_files, labels = load_video_data(main_path, label_dict)
train_gen = VideoDataGenerator(video_files, labels, batch_size=batch_size)

dataset = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([train_gen.batch_size, train_gen.max_frames, 256, 256, 3], [train_gen.batch_size, train_gen.n_classes])
)

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

    learning_rate = hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5])
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
    tuner.search(dataset, epochs=20, callbacks=[stop_early])
    best_model = tuner.get_best_models(num_models=1)[0]

    best_model.generator.save_weights("generator_weights_tuned.h5")
    best_model.generator.save("generator_model_tuned.h5")
    best_model.discriminator.save_weights("discriminator_weights_tuned.h5")

    # Also saving a plot showing the losses over epochs
    import matplotlib.pyplot as plt

    try:
        d_loss_history = [float(loss) for loss in best_model.d_loss_history]
        g_loss_history = [float(loss) for loss in best_model.g_loss_history]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_history, label='Discriminator Loss')
    plt.plot(g_loss_history, label='Generator Loss')
    plt.title('Training Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses_over_epochs.png')

else :
    print("Lets do some predictions!!")

    def generate_and_plot_images(generator, word, label_dict, latent_dim=100, num_images=10, grid_dim=(2, 5), fileName = "generated_images.png"):
        # Ensure the word is in the dictionary
        if word not in label_dict:
            print(f"Word '{word}' not found in label dictionary.")
            return

        word_label = label_dict[word]
        num_classes = max(label_dict.values()) # Calculate correct number of classes

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

word_list = ['tired', 'still']
for word in word_list:
    fileName = f"generated_images_tuned_{word}.png"
    generate_and_plot_images(generator, word, label_dict, latent_dim=latent_dim, num_images=10, grid_dim=(2, 5), fileName=fileName)

