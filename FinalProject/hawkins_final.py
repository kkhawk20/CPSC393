from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense
from tensorflow.keras.applications import ResNet50

# # Load pre-trained CNN
# base_model = ResNet50(weights='imagenet', include_top=False)

# model = Sequential([
#     TimeDistributed(base_model, input_shape=(None, frame_height, frame_width, channels)),
#     TimeDistributed(Dense(512, activation='relu')),
#     LSTM(256),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

# Reading in data
# The data is set up as classes, with subfolders representing the video representing that class

'''
1. Encoder:
    - Image Encoder (takes in frames and generates sequence of latent representations) CNN based
2. Decoder:
    - Sequence model (Takes in latent rep and feed into transformer or LSTM)

Generation:
    - Sentence encoder (transformer)
    - Frame sequence generator (sequence model from encoder)
    - Frame generator (GAN)
'''

import tensorflow as tf
import os
from pathlib import Path

class ASLVideoDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Use Path for a more robust cross-platform compatibility
        for class_folder in Path(self.root_dir).iterdir():
            if class_folder.is_dir():
                for video_folder in class_folder.iterdir():
                    if video_folder.is_dir():
                        frames = sorted(video_folder.glob('*.jpg'))
                        samples.append({
                            'label': class_folder.name,
                            'video_id': video_folder.name,
                            'frame_paths': [str(frame) for frame in frames]
                        })
        return samples

    def read_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = (img / 255.0) - 1  # Normalize to [-1, 1]
        return img

    def load_and_preprocess_video(self, sample):
        # Convert the list of paths to a tensor
        frame_paths_tensor = tf.convert_to_tensor(sample['frame_paths'], dtype=tf.string)
        # Use tf.map_fn to apply read_image function to each element in the tensor
        frames = tf.map_fn(self.read_image, frame_paths_tensor, dtype=tf.float32, back_prop=False)
        return frames, sample['label']

    def create_tf_dataset(self):
        # Use generator function for the dataset
        def gen():
            for sample in self.samples:
                yield self.load_and_preprocess_video(sample)

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
        return dataset

# Usage
root_dir = '/app/rundir/CPSC393/FinalProject/images/'
asl_dataset = ASLVideoDataset(root_dir)
tf_dataset = asl_dataset.create_tf_dataset()
tf_dataset = tf_dataset.shuffle(100).batch(4).prefetch(tf.data.experimental.AUTOTUNE)

# Example of iterating over the dataset
for frames, label in tf_dataset.take(1):
    print(frames.shape)  # Should print (num_frames, 224, 224, 3)
    print(label.numpy())  # Should print the label of the video


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense
from tensorflow.keras.applications import ResNet50

# Load pre-trained CNN
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base
base_model.trainable = False

# Model Definition
model = Sequential([
    TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
    TimeDistributed(Dense(512, activation='relu')),
    LSTM(256),
    Dense(10, activation='softmax')  # Adjust the number of classes
])

# Model compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model fitting
# Convert labels to one-hot encoding, adjust as needed for your labels
def preprocess_labels(labels):
    return tf.one_hot(tf.strings.to_hash_bucket_strong(labels, num_buckets=10), 10)

tf_dataset = tf_dataset.map(lambda x, y: (x, preprocess_labels(y)))
model.fit(tf_dataset, epochs=10)
