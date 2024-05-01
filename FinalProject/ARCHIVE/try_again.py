import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import TextVectorization, Embedding
from PIL import Image
import cv2
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Constants
TEXT_EMBEDDING_DIM = 50
NOISE_DIM = 100
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 50

# Load JSON data for glosses and bounding boxes
main_path = "/app/rundir/CPSC393/FinalProject/"
with open(main_path + 'WLASL_v0.3.json', 'r') as data_file:
    json_data = json.load(data_file)

# Prepare text vectorization and embedding
max_features = 10000
max_len = 50
embedding_dim = TEXT_EMBEDDING_DIM
text_vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=max_len)
text_embedding = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len)

# Directory containing subfolders of images, each named by class labels
image_dir = "/app/rundir/CPSC393/FinalProject/images/"

# Create a generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=(256, 256),
    batch_size=16, 
    class_mode='categorical')

# Since class_mode is 'categorical', the labels will be one-hot encoded, but we need class names for text embedding.
# Capture class labels from the training generator
class_labels = list(train_generator.class_indices.keys())
text_vectorizer.adapt(class_labels)

# Generator model
def make_generator_model():
    text_input = layers.Input(shape=(TEXT_EMBEDDING_DIM,))
    noise_input = layers.Input(shape=(NOISE_DIM,))
    x = layers.Concatenate()([text_input, noise_input])
    x = layers.Dense(8*8*256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    return models.Model(inputs=[text_input, noise_input], outputs=x)

# Discriminator model
def make_discriminator_model():
    image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    text_input = layers.Input(shape=(TEXT_EMBEDDING_DIM,))
    
    # Process text input and upscale
    ti = layers.Dense(1024, activation='relu')(text_input)
    ti = layers.Reshape((32, 32, 1))(ti)
    ti = layers.Conv2DTranspose(3, (3, 3), strides=(8, 8), padding='same')(ti)  # Ensure depth is 3

    # Concatenate along the channel axis
    combined_input = layers.Concatenate()([image_input, ti])
    
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(combined_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    return models.Model(inputs=[image_input, text_input], outputs=x)

# Loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = optimizers.Adam(2e-4, beta_1=0.5)

# Model instantiation
generator = make_generator_model()
discriminator = make_discriminator_model()

# Checkpoint directory
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Training step function
@tf.function
def train_step(images, labels):
    class_names = tf.gather(class_labels, tf.argmax(labels, axis=1))
    text_vectors = text_vectorizer(class_names)
    text_embeddings = text_embedding(text_vectors)

    noise = tf.random.normal([tf.shape(images)[0], NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([text_embeddings, noise], training=True)
        real_output = discriminator([images, text_embeddings], training=True)
        fake_output = discriminator([generated_images, text_embeddings], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Function to generate and save images
def generate_and_save_images(model, epoch, test_input, file_path='./'):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)  # Scale the images to [0, 1]
        plt.axis('off')
    plt.savefig(file_path + f'image_at_epoch_{epoch:04d}.png')
    plt.close()

# Training function
def train(epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch, label_batch in train_generator:
            train_step(image_batch, label_batch)
        print(f"Time for epoch {epoch + 1} is {time.time() - start:.2f} sec")
        if (epoch + 1) % 5 == 0:  # Save images every 5 epochs
            generate_and_save_images(generator, epoch + 1, test_input=fixed_text_embeddings, file_path=checkpoint_dir + '/')
        if (epoch + 1) % 10 == 0:  # Save model every 10 epochs
            checkpoint.save(file_prefix=checkpoint_prefix)

# Generate fixed random vector and text for viewing training progress
fixed_noise = tf.random.normal([16, NOISE_DIM])  # Example to generate 16 images
fixed_texts = ["Apple Tree"] * 16  # Example descriptions
fixed_text_vectors = text_vectorizer(fixed_texts)
fixed_text_embeddings = text_embedding(fixed_text_vectors)
test_input = [fixed_text_embeddings, fixed_noise]

# Start training
train(EPOCHS)
