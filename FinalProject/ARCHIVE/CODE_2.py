import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from PIL import Image
import glob


def get_vids(path2ajpgs):
    ids, labels = [], []
    listOfCats = os.listdir(path2ajpgs)
    if '.DS_Store' in listOfCats:
        listOfCats.remove('.DS_Store')
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        for los in listOfSubCats:
            ids.append(os.path.join(path2catg, los))
            labels.append(catg)
    return ids, labels

class VideoSequence(Sequence):
    def __init__(self, video_paths, labels, batch_size, sequence_length=16):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        video_data = [self.load_frames(path) for path in batch_x]
        return np.array(video_data), np.array(batch_y)

    def load_frames(self, path):
        frames = sorted(glob.glob(f"{path}/*.jpg"))[:self.sequence_length]
        frames = [np.array(Image.open(frame).convert('RGB')) for frame in frames]
        if len(frames) < self.sequence_length:
            frames += [np.zeros((224, 224, 3)) for _ in range(self.sequence_length - len(frames))]
        return np.stack(frames, axis=0)

def build_generator(latent_dim, text_embedding_dim, hidden_dim, num_frames):
    noise_input = layers.Input(shape=(latent_dim,))
    text_input = layers.Input(shape=(1,))
    text_embedding = layers.Embedding(2000, text_embedding_dim)(text_input)
    text_embedding = layers.Flatten()(text_embedding)
    
    combined_input = layers.Concatenate()([noise_input, text_embedding])
    x = layers.Dense(hidden_dim, activation='relu')(combined_input)
    x = layers.RepeatVector(num_frames)(x)
    x = layers.LSTM(hidden_dim, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(3 * 224 * 224, activation='tanh'))(x)
    x = layers.Reshape((num_frames, 224, 224, 3))(x)
    
    model = models.Model(inputs=[noise_input, text_input], outputs=x)
    return model

def build_discriminator():
    video_input = layers.Input(shape=(None, 224, 224, 3))
    x = layers.TimeDistributed(layers.Flatten())(video_input)
    x = layers.LSTM(512)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=video_input, outputs=x)
    return model

class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        videos, _ = data
        batch_size = tf.shape(videos)[0]

        # Train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_videos = self.generator([random_latent_vectors, tf.random.uniform((batch_size, 1), maxval=2000, dtype=tf.int32)])
        combined_videos = tf.concat([videos, generated_videos], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_videos)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            generated_videos = self.generator([random_latent_vectors, tf.random.uniform((batch_size, 1), maxval=2000, dtype=tf.int32)])
            predictions = self.discriminator(generated_videos)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

def generate_fake_videos(generator, num_samples, latent_dim, num_frames, text_embedding_dim):
    # Random noise vector and random text indices
    random_latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))
    random_text_indices = tf.random.uniform((num_samples, 1), maxval=2000, dtype=tf.int32)

    # Generate fake videos
    fake_videos = generator.predict([random_latent_vectors, random_text_indices])
    return fake_videos

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    return keras.models.load_model(filepath)

def plot_losses(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_losses.png")
    plt.show()

def main():
    # Paths and parameters
    main_path = '/app/rundir/CPSC393/FinalProject/images'
    latent_dim = 100
    num_frames = 16
    text_embedding_dim = 20
    hidden_dim = 512
    num_samples = 5

    # Build models
    generator = build_generator(latent_dim, text_embedding_dim, hidden_dim, num_frames)
    discriminator = build_discriminator()

    # Compile GAN
    gan = GAN(generator=generator, discriminator=discriminator, latent_dim=latent_dim)
    gan.compile(
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy()
    )

    # Load data and create sequence
    ids, labels = get_vids(main_path)  # Assuming get_vids is defined to load your dataset
    train_sequence = VideoSequence(ids, labels, batch_size=10, sequence_length=num_frames)

    # Train GAN
    history = gan.fit(train_sequence, epochs=20)
    plot_losses(history.history['d_loss'], history.history['g_loss'])

    # Generate fake videos
    fake_videos = generate_fake_videos(generator, num_samples, latent_dim, num_frames, text_embedding_dim)

    # Save models
    save_model(generator, 'generator_model.h5')
    save_model(discriminator, 'discriminator_model.h5')

if __name__ == "__main__":
    main()



