import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D,LSTM,  BatchNormalization, Conv2DTranspose, TimeDistributed, Conv2D, Flatten, Dense
import numpy as np
import cv2
import os
import json
import pandas as pd
from tensorflow.keras import layers, models
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (1 = INFO, 2 = WARNING, 3 = ERROR)
tf.config.optimizer.set_jit(False)  # Disable the Just-In-Time compilation optimization.

# Clear the TensorFlow session
tf.keras.backend.clear_session()


'''
Reading in JSON file and creating key for ASL Video dataset
Includes bounding boxes as well as class labels, etc. 
'''
main_path = "/app/rundir/CPSC393/FinalProject/"

batch_size = 2
num_channels = 1
num_classes = 408
image_size = 256
latent_dim = 408

# Reading in the JSON file
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

# Helper function to apply bounding box cropping
def apply_bbox(frame, bbox):
    """Apply bounding box to the frame if it's within frame boundaries."""
    y, x, h, w = bbox  # assuming bbox is [y, x, height, width]
    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
        return frame[y:y+h, x:x+w]
    else:
        print(f"Bounding box {bbox} is out of frame boundaries {frame.shape}")
        return frame  # Return full frame if bbox is out of boundaries

# Load label dictionary
label_dict_path = './labels_dict.txt'  # Update this path
label_dict = {}
with open(label_dict_path, 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        label_dict[key] = int(value)  # Ensure the number is converted to int

# Assuming the dictionary format is {'word': number, ...}
inverse_label_dict = {v: k for k, v in label_dict.items()}  # For decoding purposes, if needed

def load_and_process_frame(frame_file, video_id, bbox_df):
    print(f"Processing file: {frame_file}")
    try:
        frame = cv2.imread(frame_file)
        if frame is None:
            raise ValueError("Unable to read the image file, it may be corrupted.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Convert back to RGB
        frame = cv2.resize(frame, (256, 256))
        frame = frame / 255.0
        print("Success...")
        return frame
    except Exception as e:
        print(f"Error processing {frame_file}: {e}")
        return None

# modified for testing
def load_video_data_and_labels(directory, bbox_df, label_dict, max_frames=16, max_videos=20):
    video_data = []
    labels = []
    video_count = 0  # Counter to keep track of how many videos have been processed

    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path) or video_count >= max_videos:
            continue
        for video_id in os.listdir(category_path):
            if video_count >= max_videos:
                break  # Stop processing if the limit is reached
            video_id_path = os.path.join(category_path, video_id)
            if not os.path.isdir(video_id_path):
                continue
            frame_files = sorted([os.path.join(video_id_path, f) for f in os.listdir(video_id_path) if f.endswith('.jpg')])
            video_frames = [load_and_process_frame(frame_file, video_id, bbox_df) for frame_file in frame_files]
            video_frames = [f for f in video_frames if f is not None]  # Filter out failed loads

            # Padding or truncating the frames
            if len(video_frames) < max_frames:
                # Pad with zeros if less than max_frames
                padding = [np.zeros((256, 256, 3)) for _ in range(max_frames - len(video_frames))]
                video_frames.extend(padding)
            elif len(video_frames) > max_frames:
                # Truncate if more than max_frames
                video_frames = video_frames[:max_frames]

            if video_frames:  # Ensure there are frames
                video_data.append(np.stack(video_frames))
                labels.append(label_dict.get(category, -1))  # Get label, default to -1 if not found
            video_count += 1

    return np.array(video_data), np.array(labels)

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

'''
Creating a generator for the video data, which will load and process each set of videos (sequence of frames)
'''
class VideoDataGenerator(Sequence):
    def __init__(self, video_files, labels, batch_size, max_frames=16, n_channels=3, n_classes=408, shuffle=True):
        'Initialization'
        self.dim = (256, 256)  # Assuming all images are resized to this dimension
        self.video_files = video_files
        self.labels = labels
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.video_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_video_files_temp = [self.video_files[k] for k in indexes]
        X, y = self.__data_generation(list_video_files_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.video_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_video_files_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, self.max_frames, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, video_files in enumerate(list_video_files_temp):
            video_frames = [self.load_and_process_frame(f) for f in video_files]
            video_frames = [f for f in video_frames if f is not None]  # Filter out failed frames

            if len(video_frames) < self.max_frames:
                # Pad with zeros if less than max_frames
                padding_length = self.max_frames - len(video_frames)
                padding = np.zeros((padding_length, *self.dim, self.n_channels))
                video_frames = np.vstack((video_frames, padding))
            elif len(video_frames) > self.max_frames:
                # Truncate if more than max_frames
                video_frames = video_frames[:self.max_frames]

            X[i,] = video_frames
            y[i] = self.labels[i]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def load_and_process_frame(self, frame_file):
        'A function to load and process a single frame'
        try:
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Assuming model needs RGB inputs
            frame = cv2.resize(frame, self.dim)
            frame = frame / 255.0  # Normalize to [0, 1]
            return frame
        except Exception as e:
            print(f"Error processing {frame_file}: {e}")
            return None

main_path = "/app/rundir/CPSC393/FinalProject/images/"
video_files, labels = load_video_data(main_path, label_dict)
train_gen = VideoDataGenerator(video_files, labels, batch_size=batch_size)

# Convert generator to tf.data.Dataset
dataset = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([train_gen.batch_size, train_gen.max_frames, 256, 256, 3], [train_gen.batch_size, train_gen.n_classes])
)

'''
MODEL BUILDING
Generator (Takes in noise and label as input)
Discriminator (Takes in video frames as input)
'''
def build_generator(latent_dim, num_classes, seq_length=16, height=256, width=256, channels=3):
    noise_input = keras.Input(shape=(latent_dim,))
    label_input = keras.Input(shape=(num_classes,))
    concat_input = layers.Concatenate()([noise_input, label_input])

    # Dense layer to expand the input
    x = layers.Dense(8*8*512, activation='relu')(concat_input)
    x = layers.Reshape((8, 8, 512))(x)

    # Transposed convolutions to upscale to the desired dimensions
    x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')(x)  # Ensure this final layer reaches 256x256

    # Replicating the frame to form a sequence
    x = layers.Reshape((height, width, channels))(x)
    x = layers.Lambda(lambda x: tf.repeat(tf.expand_dims(x, axis=1), repeats=seq_length, axis=1))(x)

    model = keras.Model(inputs=[noise_input, label_input], outputs=x)
    return model

def build_discriminator(seq_length=16, height=256, width=256, channels=3):
    model = keras.Sequential([
        keras.Input(shape=(seq_length, height, width, channels)),
        TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")),
        TimeDistributed(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")),

        TimeDistributed(Flatten()),

        LSTM(128, return_sequences = False),

        Flatten(),
        Dense(1)
    ])
    return model

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')

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
        real_videos, _ = data  # Ignore labels from the dataset for discriminator training
        batch_size = tf.shape(real_videos)[0]

        # Generating random labels for fake video generation
        random_labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
        label_one_hot = tf.one_hot(random_labels, depth=num_classes)

        # Generating random latent vectors
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train Discriminator
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_videos = self.generator([random_latent_vectors, label_one_hot], training=True)
            real_predictions = self.discriminator(real_videos, training=True)
            fake_predictions = self.discriminator(generated_videos, training=True)
            d_loss_real = self.d_loss_fn(tf.ones_like(real_predictions), real_predictions)
            d_loss_fake = self.d_loss_fn(tf.zeros_like(fake_predictions), fake_predictions)
            d_loss = d_loss_real + d_loss_fake
            g_loss = self.g_loss_fn(fake_predictions)

        # Gradient computation
        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)

        # Apply gradients
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Custom loss functions for the discriminator and generator
def discriminator_loss(real_output, fake_output):
    smooth_positive_labels = 0.9  # Soft label for "real"
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output) * smooth_positive_labels, real_output, from_logits=True)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


'''
TRAINING MODEL LOGIC & PLAN! 
'''
# Build generator
try:
    generator = build_generator(latent_dim=latent_dim, num_classes=num_classes)
    print("Generator built successfully.")
except Exception as e:
    print("Failed to build generator:", e)

# Build discriminator
try:
    discriminator = build_discriminator()
    print("Discriminator built successfully.")
except Exception as e:
    print("Failed to build discriminator:", e)

# Instantiate and compile the Conditional GAN
try:
    cond_gan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        g_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss
    )
    print("Conditional GAN built and compiled successfully.")
except Exception as e:
    print("Failed to instantiate or compile Conditional GAN:", e)

print(" ~~~~~~~~~~ Training the model ~~~~~~~~~~~~~")

# Ensure the generator can process an input
test_noise = tf.random.normal([1, latent_dim])
test_label = tf.one_hot([3], depth=num_classes)  # Example label
_ = generator([test_noise, test_label])

# Ensure the discriminator can handle its expected input shape
dummy_video = tf.random.normal([batch_size, 16, 256, 256, 3])  # Adjust based on your discriminator's input requirements
_ = discriminator(dummy_video)

cond_gan.train_step((dummy_video, [test_noise, test_label]))

# Setup model checkpointing
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'gan_checkpoint.h5', save_weights_only=True, 
    save_best_only=True, monitor='g_loss', mode='min'
)

# Early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='g_loss', patience=20, restore_best_weights=True,
    mode='min'
)

# Example of setting up shapes for noise and labels
noise_shape = (None, latent_dim)
label_shape = (None, num_classes)

# Build the model manually for these input shapes
cond_gan.build([noise_shape, label_shape])

# Now call summary
cond_gan.summary()

# Check build status and perform a training step
print("Conditional GAN built:", cond_gan.built)
cond_gan.train_step((dummy_video, None))  # This should ideally not be needed after build() but just to confirm

retrain = True
# Fit the model with actual data
if cond_gan.built and retrain:
    print("Model is built")
    cond_gan.fit(dataset, epochs=1000, 
                 callbacks=[checkpoint_callback, early_stopping_callback])
    generator.save_weights('generator_weights.h5')
    discriminator.save_weights('discriminator_weights.h5')
    generator.save('generator_model.h5')
else:
    print("Model is still not built or you dont want to retrain it...")

'''
MAKING GENERATED IMAGES!!!
'''
print("~~~~~~~~~~ Making images ~~~~~~~~~~")

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

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
generator = load_model("generator_model.h5", compile=False)

word_list = ['tired', 'still']
for word in word_list:
    fileName = f"generated_images_{word}.png"
    generate_and_plot_images(generator, word, label_dict, latent_dim=latent_dim, num_images=10, grid_dim=(2, 5), fileName=fileName)
