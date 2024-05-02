import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2DTranspose, TimeDistributed, Conv2D, Flatten, Dense
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
latent_dim = 100

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

# def load_video_data_and_labels(directory, bbox_df, label_dict):
#     video_data = []
#     labels = []
#     for category in os.listdir(directory):
#         category_path = os.path.join(directory, category)
#         if not os.path.isdir(category_path):
#             continue
#         for video_id in os.listdir(category_path):
#             video_id_path = os.path.join(category_path, video_id)
#             if not os.path.isdir(video_id_path):
#                 continue
#             frame_files = sorted([os.path.join(video_id_path, f) for f in os.listdir(video_id_path) if f.endswith('.jpg')])
#             video_frames = [load_and_process_frame(frame_file, video_id, bbox_df) for frame_file in frame_files]
#             video_frames = [f for f in video_frames if f is not None]  # Filter out failed loads
#             if video_frames:  # Ensure there are frames
#                 video_data.append(np.stack(video_frames))
#                 labels.append(label_dict.get(category, -1))
#     return np.array(video_data, dtype=object), np.array(labels)

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

def build_generator(latent_dim, seq_length=16, height=256, width=256, channels=3):
    model = keras.Sequential([
        # Start from a latent dimension
        keras.Input(shape=(latent_dim,)),
        # First Dense layer to create a suitable number of features
        layers.Dense(8 * 8 * 256, activation="relu"),
        layers.Reshape((8, 8, 256)),  # Reshape to small spatial dimensions but with many features

        # Use Conv2DTranspose to upscale to the desired spatial dimensions
        layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),

        # Final Conv2DTranspose to get to the correct channel size, still single frame
        layers.Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', activation='sigmoid'),

        # Reshape output to introduce the sequence dimension
        layers.Reshape((height, width, channels)),
        # Replicate this frame to form a sequence of identical frames
        layers.Lambda(lambda x: tf.repeat(tf.expand_dims(x, axis=1), repeats=seq_length, axis=1))
    ])
    return model

def build_discriminator(seq_length=16, height=256, width=256, channels=3):
    model = keras.Sequential([
        keras.Input(shape=(seq_length, height, width, channels)),
        TimeDistributed(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")),
        ConvLSTM2D(64, (3, 3), return_sequences=True, padding="same"),
        TimeDistributed(Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")),
        ConvLSTM2D(128, (3, 3), return_sequences=False, padding="same"),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_videos, labels = data  # real_videos should be [batch_size, seq_length, height, width, channels]
        batch_size = tf.shape(real_videos)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Generate fake videos
        generated_videos = self.generator(random_latent_vectors)

        # Combine real and generated videos for the discriminator
        combined_videos = tf.concat([generated_videos, real_videos], axis=0)
        combined_labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            disc_predictions = self.discriminator(combined_videos)
            d_loss = self.loss_fn(combined_labels, disc_predictions)

        disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

        # Misleading labels for generator training (trying to fool the discriminator)
        misleading_labels = tf.ones((batch_size, 1))

        # Train generator (only through generator gradients)
        with tf.GradientTape() as gen_tape:
            fake_predictions = self.discriminator(generated_videos)
            g_loss = self.loss_fn(misleading_labels, fake_predictions)

        gen_grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}
    
# Build generator
try:
    generator = build_generator(latent_dim=100)
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
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
        loss_fn=keras.losses.BinaryCrossentropy()
    )
    print("Conditional GAN built and compiled successfully.")
except Exception as e:
    print("Failed to instantiate or compile Conditional GAN:", e)

print(" ~~~~~~~~~~ Training the model ~~~~~~~~~~~~~")
# Fit the model
cond_gan.fit(dataset, epochs=30)
cond_gan.save('generator_model.h5')

'''
MAKING GENERATED IMAGES!!!
'''
print("~~~~~~~~~~ Making images ~~~~~~~~~~")
# Load the saved generator model
def generate_and_plot_images(generator, word, label_dict, num_images=5, grid_dim=(1, 5)):
    """
    Generate and plot a grid of images for a specific word.
    Args:
    - generator: The trained generator model.
    - word: The word to generate signs for.
    - label_dict: Dictionary of word to label mappings.
    - num_images: Number of images to generate.
    - grid_dim: Tuple indicating grid dimensions (rows, columns).
    """
    if word not in label_dict:
        print(f"Word '{word}' not found in label dictionary.")
        return

    word_label = label_dict[word]
    num_classes = max(label_dict.values()) + 1  # Assuming labels start from 0
    label_one_hot = tf.one_hot([word_label] * num_images, depth=num_classes)
    noise = tf.random.normal([num_images, 100])  # Assuming noise dimension is 100

    # Generate images
    inputs = tf.concat([noise, label_one_hot], axis=1)
    images = generator.predict(inputs)

    # Rescale images to [0, 255]
    images = (images * 255).astype(np.uint8)

    # Plot images in a grid
    fig, axes = plt.subplots(grid_dim[0], grid_dim[1], figsize=(15, 3))
    axes = axes.flatten() if num_images > 1 else [axes]

    for img, ax in zip(images, axes):
        ax.imshow(img.reshape(256, 256, 3))  # Adjust the shape based on your model's output
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('generated_images.png')

# Example usage
generator = load_model("generator_model.h5")  # Load your trained generator model
word = 'able'  # Example ASL word
generate_and_plot_images(generator, word, label_dict, num_images=5, grid_dim=(1, 5))