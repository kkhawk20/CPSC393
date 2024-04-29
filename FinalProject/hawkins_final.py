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

import tensorflow as tf
from pathlib import Path

class ASLVideoDataset:
    def __init__(self, root_dir, max_frames = 30):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.samples = []
        self.labels = set()  # Set to collect unique labels
        self._load_samples()

    def _load_samples(self):
        for class_folder in Path(self.root_dir).iterdir():
            if class_folder.is_dir():
                self.labels.add(class_folder.name)  # Add label to the set
                for video_folder in class_folder.iterdir():
                    if video_folder.is_dir():
                        frames = sorted(video_folder.glob('*.jpg'))
                        self.samples.append({
                            'label': class_folder.name,
                            'video_id': video_folder.name,
                            'frame_paths': [str(frame) for frame in frames]
                        })

    def read_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = (img / 255.0) - 1  # Normalize to [-1, 1]
        return img

    def load_and_preprocess_video(self, sample):
        frames = [self.read_image(frame) for frame in sample['frame_paths']]
        num_frames = len(frames)
        # Pad frames to max_frames
        frames += [tf.zeros((224, 224, 3))] * (self.max_frames - num_frames)
        frames = tf.stack(frames)
        return frames, sample['label']

    def create_tf_dataset(self):
        def gen():
            for sample in self.samples:
                yield self.load_and_preprocess_video(sample)
        return tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.float32, tf.string),
            output_shapes=((self.max_frames, 224, 224, 3), ())
        )

# Creating the datast
root_dir = '/app/rundir/CPSC393/FinalProject/images/'
asl_dataset = ASLVideoDataset(root_dir, max_frames = 30)
tf_dataset = asl_dataset.create_tf_dataset()
tf_dataset = tf_dataset.shuffle(100).padded_batch(4, padded_shapes=([None, 224, 224, 3], [])).prefetch(tf.data.experimental.AUTOTUNE)
num_classes = len(asl_dataset.labels) # We now have the number of unique classes

# Example of iterating over the dataset
for frames, label in tf_dataset.take(1):
    print(frames.shape)  # Should print (batch_size, num_frames, 224, 224, 3)
    print(label.numpy())  # Should print the label of the videos




# Model building

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, Input
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes):
    input_shape = (30, 224, 224, 3)

    cnn_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(244, 244, 3))
    cnn_base.trainable = False

    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))),
        TimeDistributed(Flatten()),
        LSTM(50),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.build(input_shape=(None,) + input_shape)  # Build the model with the specified input shape
    return model

# Create a mapping from label strings to integers
label_to_index = {label: idx for idx, label in enumerate(sorted(asl_dataset.labels))}

def map_labels(frames, label):
    # Convert label to its corresponding index using TensorFlow operations within tf.py_function
    def decode_label(label):
        return label_to_index[tf.compat.as_text(label.numpy())]
    label_index = tf.py_function(decode_label, [label], tf.int32)
    return frames, label_index

tf_dataset = asl_dataset.create_tf_dataset()
tf_dataset = tf_dataset.map(map_labels).shuffle(100).batch(4).prefetch(tf.data.experimental.AUTOTUNE)

model = build_model(num_classes)  # specify the shape details
model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
model.summary()

# Setup training with callbacks for better monitoring and checkpointing
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os

checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5)

# Create a TensorBoard instance to visually monitor training
tensorboard_callback = TensorBoard(log_dir='./logs')

for frames, labels in tf_dataset.take(1):
    print("Frames shape:", frames.shape)  # Should show (batch_size, num_frames, 224, 224, 3)
    print("Labels type:", labels.dtype)   # Should show tf.int64 or tf.int32

for inputs, labels in tf_dataset.take(1):
    print("Input shape:", inputs.shape)
    print("Label shape:", labels.shape)
    predictions = model(inputs, training=False)  # Run a forward pass
    print("Predictions shape:", predictions.shape)

num_classes = len(asl_dataset.labels)  # Ensure this reflects the actual number of unique classes correctly.
print("Number of classes:", num_classes)

for inputs, labels in tf_dataset.take(3):  # Check several batches to ensure consistency
    print("Batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)

try:
    history = model.fit(tf_dataset, epochs=10, steps_per_epoch=len(asl_dataset.samples) // 4)
except ValueError as e:
    print("Error during training:", e)
    print("Investigating further...")
    # Add more diagnostics here if needed

# history = model.fit(
#     tf_dataset,
#     epochs=10,
#     steps_per_epoch=len(asl_dataset.samples) // 4
# )

# callbacks=[cp_callback, tensorboard_callback]

import numpy as np
import tensorflow as tf

# Create a simple synthetic dataset
test_inputs = np.random.random((10, 30, 224, 224, 3)).astype(np.float32)  # 10 videos
test_labels = np.random.randint(0, 409, size=(10,)).astype(np.int32)  # 10 labels

test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
test_dataset = test_dataset.batch(4)

# Try fitting the model on this synthetic dataset
model.fit(test_dataset, epochs=1)


