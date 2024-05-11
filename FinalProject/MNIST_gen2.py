import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Flatten, Dense, Reshape, UpSampling2D, Concatenate, Input
from tensorflow.keras.applications import ResNet50
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
batch_size = 64
num_classes = 26  # Ensure this matches your dataset
latent_dim = 100

train_df = pd.read_csv("./sign_mnist_train.csv")
test_df = pd.read_csv("./sign_mnist_test.csv")

# Separating X and Y
y_train = train_df.pop('label')
y_test = test_df.pop('label')

# Rescale data to be 0-1 instead of 0-255
trainX = train_df.values.reshape(-1, 28, 28, 1) / 255.0
testX = test_df.values.reshape(-1, 28, 28, 1) / 255.0

# Change the labels to be in the correct format
lb = LabelBinarizer()
trainY = lb.fit_transform(y_train)
testY = lb.transform(y_test)

# Adjusting labels to match num_classes
trainY = np.pad(trainY, ((0, 0), (0, num_classes - trainY.shape[1])), 'constant')
testY = np.pad(testY, ((0, 0), (0, num_classes - testY.shape[1])), 'constant')

# Visualize some images
f, ax = plt.subplots(2, 5)
f.set_size_inches(10, 10)
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(trainX[j + i * 5].reshape(28, 28), cmap="gray")
    plt.tight_layout()    
plt.savefig("ASL_MNIST_Images.png")

# Use tf.data to create the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

def build_generator_with_resnet(latent_dim, num_classes):
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(num_classes,))

    x = Concatenate()([noise_input, label_input])
    x = Dense(7 * 7 * 256, activation='relu')(x)
    x = Reshape((7, 7, 256))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(3, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)
    
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(114, 114, 3))
    resnet_base.trainable = False  # Freeze the ResNet layers
    
    x = resnet_base(x)
    x = Flatten()(x)
    output = Dense(28 * 28, activation='sigmoid')(x)
    output = Reshape((28, 28, 1))(output)
    
    model = keras.Model([noise_input, label_input], output)
    return model

def build_discriminator(num_classes):
    image_input = Input(shape=(28, 28, 1))
    label_input = Input(shape=(num_classes,))

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(image_input)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Flatten()(x)
    
    x = Concatenate()([x, label_input])
    x = Dense(1, activation='sigmoid')(x)
    
    model = keras.Model([image_input, label_input], x)
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, num_classes):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def compile(self, d_optimizer, g_optimizer):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = discriminator_loss
        self.g_loss_fn = generator_loss
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        images, labels = data
        batch_size = tf.shape(images)[0]

        # Create random noise and one-hot labels
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_labels = tf.one_hot(tf.argmax(labels, axis=1), self.num_classes)

        # Debugging batch sizes
        print(f"Batch size: {batch_size}")

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = self.generator([random_latent_vectors, fake_labels], training=True)
            real_output = self.discriminator([images, labels], training=True)
            fake_output = self.discriminator([generated_images, fake_labels], training=True)

            # Debugging shapes
            print(f"Real images shape: {images.shape}, Generated images shape: {generated_images.shape}")
            print(f"Real output shape: {real_output.shape}, Fake output shape: {fake_output.shape}")

            d_loss = self.d_loss_fn(real_output, fake_output)
            g_loss = self.g_loss_fn(fake_output)

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_weights)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# Initialize and compile the GAN
discriminator = build_discriminator(num_classes)
generator = build_generator_with_resnet(latent_dim, num_classes)
gan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim, num_classes=num_classes)
gan.compile(
    d_optimizer=keras.optimizers.Adam(0.0001),
    g_optimizer=keras.optimizers.Adam(0.0001)
)

# Filter out batches that are smaller than the required batch size
def filter_incomplete_batches(dataset, batch_size):
    return dataset.filter(lambda x, y: tf.shape(x)[0] == batch_size)

train_dataset = filter_incomplete_batches(train_dataset, batch_size)

# Debugging the dataset before training
for batch in train_dataset.take(1):
    print(f"Sample batch shapes - images: {batch[0].shape}, labels: {batch[1].shape}")

# Train the GAN
history = gan.fit(train_dataset, epochs=50)

# Plotting the training losses
def plot_training_losses(history, file_name="training_losses.png"):
    d_loss = history.history['d_loss']
    g_loss = history.history['g_loss']
    epochs = range(1, len(d_loss) + 1)

    plt.figure()
    plt.plot(epochs, d_loss, 'r', label='Discriminator Loss')
    plt.plot(epochs, g_loss, 'b', label='Generator Loss')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_name)
    plt.show()

plot_training_losses(history, file_name="training_losses.png")

# Function to generate and plot images
def generate_and_plot_images(generator, word, label_map, latent_dim, num_classes, grid_dim=(2, 5), file_name="generated_images.png"):
    labels = [label_map[letter] for letter in word.lower() if letter in label_map]

    noise = tf.random.normal([len(labels), latent_dim])
    label_one_hot = tf.one_hot(labels, num_classes)

    images = generator.predict([noise, label_one_hot])
    images = (images * 255).astype(np.uint8)

    fig, axes = plt.subplots(grid_dim[0], grid_dim[1], figsize=(15, 10))
    for img, ax, letter in zip(images, axes.flatten(), word):
        ax.imshow(img[:, :, 0], cmap='gray')
        ax.set_title(f"Sign for '{letter}'")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(file_name)

# Example label dictionary
label_dict = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
    'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
    't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25
}

# Generate and plot images for a word
generate_and_plot_images(generator, "hello", label_dict, 
                         latent_dim=latent_dim, num_classes=num_classes, 
                         grid_dim=(1, 5), file_name="generated_images_hello.png")
