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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow WARNING messages

'''
Reading in JSON file and creating key for ASL Video dataset
Includes bounding boxes as well as class labels, etc. 
'''
main_path = "/app/rundir/CPSC393/FinalProject/"

batch_size = 64
num_channels = 1
num_classes = 2000
image_size = 28
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
def apply_bbox(image, video_id):
    if video_id in bbox_df.index and bbox_df.loc[video_id]['bbox'] is not None:
        bbox = bbox_df.loc[video_id]['bbox']
        image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3], bbox[2])
    return image

# Load label dictionary
label_dict_path = './labels_dict.txt'  # Update this path
label_dict = {}
with open(label_dict_path, 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        label_dict[key] = int(value)  # Ensure the number is converted to int

# Assuming the dictionary format is {'word': number, ...}
inverse_label_dict = {v: k for k, v in label_dict.items()}  # For decoding purposes, if needed

def load_video_data_and_labels(directory, bbox_df, label_dict):
    video_data = []
    labels = []
    categories = os.listdir(directory)
    for category in categories:
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path):  # Skip non-directory files
            continue
        video_ids = os.listdir(category_path)
        for video_id in video_ids:
            video_id_path = os.path.join(category_path, video_id)
            if not os.path.isdir(video_id_path):  # Skip non-directory files
                continue
            frame_files = sorted([os.path.join(video_id_path, f) 
                                  for f in os.listdir(video_id_path) 
                                  if f.endswith('.jpg')])
            video_frames = []
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if video_id in bbox_df.index and bbox_df.loc[video_id]['bbox'] is not None:
                    bbox = bbox_df.loc[video_id]['bbox']
                    frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                frame = cv2.resize(frame, (256, 256))
                frame = frame / 255.0
                video_frames.append(frame)
            video_data.append(np.stack(video_frames))
            labels.append(label_dict.get(category, -1))
    return np.array(video_data), np.array(labels)

video_data, labels= load_video_data_and_labels(main_path + 'images/', bbox_df, label_dict)
dataset = tf.data.Dataset.from_tensor_slices((video_data, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Example of preparing one-hot labels
num_classes = len(label_dict)  # Total number of classes
labels_one_hot = tf.one_hot(labels, depth=num_classes)

# Prepare dataset
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

# Fit the model
cond_gan.fit(dataset, epochs=30)

'''
-------------------------------------------------------------------

from PIL import Image

# We first extract the trained generator from our Conditiona GAN.
trained_gen = cond_gan.generator

#### Re-run cell to get new image, change my_number to sample different digits ####

# choose a number
my_number = 4

# Sample Noise

# get noise array same shape as generator is expecting
my_noise = tf.random.normal(shape=(1, generator_in_channels-10))

# Sample Label

# take number and one hot encode it
my_label = keras.utils.to_categorical([my_number], num_classes)

# convert this one-hot-encoded array into a 32-bit float
my_label = tf.cast(my_label, tf.float32)

# concatenate noise and label together
my_input = tf.concat([my_noise, my_label], 1)


# make prediction using trained generator
fake_num = trained_gen.predict(my_input, verbose = False)

# re-scale output
fake_num *= 255.0

# case to 8 bit int
converted_image = fake_num.astype(np.uint8)

# resize into an actual image shape
converted_image = tf.image.resize(converted_image, (96, 96)).numpy().astype(np.uint8)

# get rid of batch size dimension
converted_image = np.squeeze(converted_image)

# plot
plt.imshow(converted_image, interpolation='nearest', cmap = 'gray')
plt.axis('off') # turn off axes
plt.show()

'''
# -------------------------------------------------------------------
'''


import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

model = tf.keras.models.load_model('model.h5')

# Mapping labels from label dictionary file
labels_dict = {}
inverse_labels_dict = {}
with open('labels_dict.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        value = int(value)
        labels_dict[key] = value
        inverse_labels_dict[value] = key

def predict_and_visualize(video_path, model, bbox_df, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    print("Video successfully opened.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read or error in fetching frame.")
            break

        print(f"Processing frame {frame_count + 1}")

        # Convert frame to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_id = os.path.basename(video_path).split('.')[0]
        bbox = None
        
        if video_id in bbox_df.index and bbox_df.loc[video_id]['bbox'] is not None:
            bbox = bbox_df.loc[video_id]['bbox']
            print(f"Original BBox for video_id {video_id}: {bbox}")

            # Calculate the reduced bounding box
            reduction_ratio = 0.8  # 40% reduction
            new_width = bbox[2] * reduction_ratio
            new_height = bbox[3] * reduction_ratio
            new_x = bbox[0] + (bbox[2] - new_width) / 2
            new_y = bbox[1] + (bbox[3] - new_height) / 2
            bbox = [int(new_x), int(new_y), int(new_width), int(new_height)]

            print(f"Reduced BBox: {bbox}")

            frame_pil = frame_pil.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Process image for the model
        frame_processed = keras_image.img_to_array(frame_pil)
        frame_processed = tf.image.resize(frame_processed, [256, 256])
        frame_processed = np.expand_dims(frame_processed, axis=0)  # Add batch dimension
        frame_processed = np.copy(frame_processed)  # Ensure the array is writable
        frame_processed /= 255.0  # Normalize to [0,1]

        # Predict using the model
        prediction = model.predict(frame_processed)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_label_name = inverse_labels_dict[predicted_label[0]]
        print(f"Predicted label: {predicted_label_name}")

        # Annotate and save frame
        if bbox:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label_name, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the frame to a file
        frame_output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_output_path, frame)
        print(f"Frame {frame_count + 1} saved at {frame_output_path}")
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed. Frames saved to:", output_dir)

# Ensure the video path and output directory are correctly specified
video_path = './videos/a/01610.mp4'  # Make sure the file extension is specified if needed
output_dir = './output_frames_a_test'
predict_and_visualize(video_path, model, bbox_df, output_dir)

'''