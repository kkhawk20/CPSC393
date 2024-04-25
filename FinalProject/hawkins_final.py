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

import os
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for word_folder in os.listdir(self.root_dir):
            word_path = os.path.join(self.root_dir, word_folder)
            for video_folder in os.listdir(word_path):
                video_path = os.path.join(word_path, video_folder)
                frames = sorted(os.listdir(video_path))
                samples.append({
                    'word': word_folder,
                    'video_id': video_folder,
                    'frame_paths': [os.path.join(video_path, frame) for frame in frames]
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = [read_image(path) for path in sample['frame_paths']]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        return sample['word'], sample['video_id'], frames

# Define transforms
transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset
asl_dataset = ASLDataset('/app/rundir/CPSC393/FinalProject/images/', transform=transforms)

# Create a DataLoader
data_loader = DataLoader(asl_dataset, batch_size=4, shuffle=True)

# Model definition and plan 
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
