import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2DTranspose, TimeDistributed, Conv2D, Flatten, Dense
import numpy as np
import cv2
import os
import json
import pandas as pd
from tensorflow.keras import layers
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (1 = INFO, 2 = WARNING, 3 = ERROR)

'''
Reading in JSON file and creating key for ASL Video dataset
Includes bounding boxes as well as class labels, etc. 
'''
main_path = "/app/rundir/CPSC393/FinalProject/"

batch_size = 8
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

def verify_images(directory):
    corrupted_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                if image is None:
                    corrupted_files.append(file_path)
    return corrupted_files

corrupted_files = verify_images('/path/to/your/dataset')
print("Corrupted JPEG files:", corrupted_files)

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

def load_video_data_and_labels(directory, bbox_df, label_dict):
    video_data = []
    labels = []
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path):
            continue
        for video_id in os.listdir(category_path):
            video_id_path = os.path.join(category_path, video_id)
            if not os.path.isdir(video_id_path):
                continue
            frame_files = sorted([os.path.join(video_id_path, f) for f in os.listdir(video_id_path) if f.endswith('.jpg')])
            video_frames = [load_and_process_frame(frame_file, video_id, bbox_df) for frame_file in frame_files]
            video_frames = [f for f in video_frames if f is not None]  # Filter out failed loads
            if video_frames:  # Ensure there are frames
                video_data.append(np.stack(video_frames))
                labels.append(label_dict.get(category, -1))
    return np.array(video_data, dtype=object), np.array(labels)

print("1. Loading in images and labels...")
video_data, labels= load_video_data_and_labels(main_path + 'images/', bbox_df, label_dict)

print("2. Creating a ragged dataset")
video_data = tf.ragged.constant(video_data, dtype=tf.float32)

print("3. Creating labels")
labels = tf.constant(labels, dtype=tf.int32)

print("4. Loading dataset creation...")
dataset = tf.data.Dataset.from_tensor_slices((video_data, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Example of preparing one-hot labels
print("5. Grabbing number of classes")
num_classes = len(label_dict)  # Total number of classes
labels_one_hot = tf.one_hot(labels, depth=num_classes)
print(num_classes)

# Prepare dataset
print("6. Creating the video_dataset")
video_dataset = tf.data.Dataset.from_tensor_slices((video_data, labels_one_hot))
video_dataset = video_dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {video_data.shape}")
print(f"Shape of training labels: {labels.shape}")

def build_generator(latent_dim, seq_length=16, frame_shape=(256, 256, 3)):
    model = keras.Sequential([
        keras.Input(shape=(latent_dim,)),
        layers.Dense(1024, activation="relu"),
        layers.Reshape((1, 1, 1024)),  # Start with a single spatial dimension
        Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="valid", activation="relu"),
        BatchNormalization(),
        Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="valid", activation="relu"),
        BatchNormalization(),
        Conv2DTranspose(128, (5, 5), strides=(4, 4), padding="valid", activation="relu"),
        BatchNormalization(),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
        ConvLSTM2D(64, (3, 3), return_sequences=True, padding='same'),
        TimeDistributed(Conv2D(3, (3, 3), activation='sigmoid', padding='same')),
        layers.Reshape((seq_length, frame_shape[0], frame_shape[1], frame_shape[2]))
    ])
    return model

def build_discriminator(frame_shape=(256, 256, 3)):
    seq_length = frame_shape[0]
    model = keras.Sequential([
        keras.Input(shape=(seq_length, frame_shape[1], frame_shape[2], 3)),
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
        # Unpack the data tuple
        real_videos, labels = data

        # Random noise
        random_latent_vectors = tf.random.normal(shape=(tf.shape(real_videos)[0], self.latent_dim))

        # Generate fake videos
        generated_videos = self.generator(random_latent_vectors)

        # Combine them with real videos for the discriminator
        combined_videos = tf.concat([generated_videos, real_videos], axis=0)
        labels = tf.concat([tf.ones((tf.shape(real_videos)[0], 1)), tf.zeros((tf.shape(real_videos)[0], 1))], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_videos)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train the generator
        misleading_labels = tf.ones((tf.shape(real_videos)[0], 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Instantiate and compile the model
generator = build_generator(latent_dim)
discriminator = build_discriminator()

cond_gan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    loss_fn=keras.losses.BinaryCrossentropy()
)

print("7. Training the model...")
# Fit the model
cond_gan.fit(dataset, epochs=30)
cond_gan.save('generator_model.h5')

'''
MAKING GENERATED IMAGES!!!
'''
print("8. Making images...")
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