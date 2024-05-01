"""
IMPORTS
"""
import os
import json
import torch
import glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device for Torch operations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1")
main_path = "/app/rundir/CPSC393/FinalProject/"

"""
Utility Functions
"""

def loading_data():
    def get_video_ids(json_list):
        video_ids = []
        for item in json_list:
            video_id = item.get('video_id', None)
            if video_id and os.path.exists(f"{main_path}/videos_raw/{video_id}.mp4"):
                video_ids.append(video_id)
        return video_ids

    wlasl_df = pd.read_json(main_path + "WLASL_v0.3.json")

    features_df = pd.DataFrame(columns=['gloss', 'video_id'])
    for row in wlasl_df.iterrows():
        ids = get_video_ids(row[1][1])
        word = [row[1][0]] * len(ids)
        df = pd.DataFrame(list(zip(word, ids)), columns=features_df.columns)
        features_df = pd.concat([features_df, df], ignore_index=True)
    return features_df

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

def create_dict():
    sub_folder_jpg = 'images'
    path2ajpgs = os.path.join(main_path, sub_folder_jpg)
    all_vids, all_labels = get_vids(path2ajpgs)
    labels_dict = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}
    with open('labels_dict.txt', 'w') as file:
        for key, value in labels_dict.items():
            file.write(f"{key}: {value}\n")
    return labels_dict

def split_data(unique_ids, unique_labels):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    train_indx, test_indx = next(sss.split(unique_ids, unique_labels))
    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]
    test_ids = [unique_ids[ind] for ind in test_indx]
    test_labels = [unique_labels[ind] for ind in test_indx]
    return train_ids, train_labels, test_ids, test_labels

"""
DEFINING THE DATASET INITIALIZER AND CREATOR
"""

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform, sequence_length=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = sorted(glob.glob(f"{self.video_paths[idx]}/*.jpg"))[:self.sequence_length]
        processed_frames = []
        for frame in frames:
            try:
                processed_frame = self.transform(Image.open(frame).convert('RGB'))
                processed_frames.append(processed_frame)
            except (IOError, OSError) as e:
                # Replace with a zero tensor if the frame is corrupted or cannot be loaded
                processed_frames.append(torch.zeros(3, 224, 224))
        # Ensure we always return a sequence of the desired length
        while len(processed_frames) < self.sequence_length:
            processed_frames.append(torch.zeros(3, 224, 224))
        frames_tensor = torch.stack(processed_frames)
        return frames_tensor, self.labels[idx]

    
def create_dataset(ids, labels, transform, sequence_length=16):
    return VideoDataset(ids, labels, transform, sequence_length)

"""
Model Definitions
"""

class Generator(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim, hidden_dim, num_frames):
        super(Generator, self).__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.text_embedding_dim = text_embedding_dim
        self.text_embedding = nn.Embedding(num_embeddings=2000, embedding_dim=text_embedding_dim)
        output_dim = num_frames * 3 * 224 * 224
        self.lstm = nn.LSTM(latent_dim + text_embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, noise, text_indices):
        text_embeddings = self.text_embedding(text_indices)
        print("Generator Shape of text_embeddings:", text_embeddings.shape)
        combined = torch.cat((noise, text_embeddings), dim=-1)
        print("Generator Shape of combined:", combined.shape)
        output, _ = self.lstm(combined)
        print("Generator Shape of LSTM output:", output.shape)
        video_frames = self.fc(output)
        print("Generator Shape of video_frames before view:", video_frames.shape)
        video_frames = video_frames.view(-1, self.num_frames, 3, 224, 224)
        print("Generator Shape of video_frames after view:", video_frames.shape)
        return video_frames

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(3 * 224 * 224, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, frames):
        frames_flat = frames.view(frames.size(0), frames.size(1), -1)
        print("Discriminator Shape of frames_flat:", frames_flat.shape)
        lstm_out, _ = self.lstm(frames_flat)
        last_hidden = lstm_out[:, -1, :]
        print("Discriminator Shape of last hidden state:", last_hidden.shape)
        validity = self.fc(last_hidden)
        print("Discriminator Shape of validity scores:", validity.shape)
        return torch.sigmoid(validity).view(-1, 1)

def sliding_window_sequences(sequence, window_size):
    return [sequence[i:i+window_size] for i in range(len(sequence) - window_size + 1)]

"""
Optimizer and Loss Function Definitions
"""

generator = Generator(latent_dim=100, text_embedding_dim=20, hidden_dim=512, num_frames=16).to(device)
discriminator = Discriminator(hidden_dim=512).to(device)
optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

"""
Plotting, Training, Saving, and Generating Functions
"""

# Methods to check reshaping possibilities
def can_reshape(tensor, shape):
    """ Check if the tensor can be reshaped to the desired shape """
    required_elements = 1
    for dim in shape:
        required_elements *= dim
    return required_elements == tensor.numel()

def safe_reshape(tensor, shape):
    """ Safely reshape tensor, catching errors and providing debugging information """
    if can_reshape(tensor, shape):
        try:
            return tensor.view(shape)
        except RuntimeError as e:
            print(f"Failed to reshape tensor from {tensor.shape} to {shape}: {str(e)}")
            raise
    else:
        error_message = f"Cannot reshape tensor of shape {tensor.shape} to {shape} - element mismatch"
        print(error_message)
        raise ValueError(error_message)

def train(generator, discriminator, dataloader, val_dataloader, optimizerG, optimizerD, criterion, num_embeddings, num_epochs, num_frames, latent_dim, window_size, device):
    d_losses, g_losses, val_losses = [], [], []
    for epoch in range(num_epochs):
        print(f"Training Epoch: {epoch + 1}")
        for i, (videos, _) in enumerate(dataloader):
            videos = videos.to(device)
            batch_size = videos.size(0)  # This should be the size of batches throughout

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optimizerD.zero_grad()
            real_loss = criterion(discriminator(videos), real_labels)

            noise = torch.randn(batch_size, num_frames, latent_dim).to(device)
            text_indices = torch.randint(0, num_embeddings, (batch_size, 1)).expand(batch_size, num_frames).to(device)

            fake_videos = generator(noise, text_indices)
            print(f"Generator Output Shape (before reshape): {fake_videos.shape}")

            # Validate total elements
            total_elements = fake_videos.numel()
            expected_elements = batch_size * num_frames * 3 * 224 * 224
            if total_elements != expected_elements:
                raise ValueError(f"Expected total elements {expected_elements}, but got {total_elements}")

            # Reshape operation
            try:
                fake_videos = fake_videos.view(batch_size, num_frames, 3, 224, 224)
            except RuntimeError as e:
                print(f"Error in reshaping: {e}")
                # Handle error appropriately, possibly skip this batch
                continue

            # try:
            #     fake_videos = safe_reshape(fake_videos, (batch_size, num_frames, 3, 224, 224))
            #     fake_videos_flat = safe_reshape(fake_videos, (batch_size, num_frames, 3 * 224 * 224))
            # except ValueError as e:
            #     print(f"Failed to reshape fake videos: {str(e)}")
            #     continue

            fake_loss = criterion(discriminator(fake_videos_flat), fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            optimizerG.zero_grad()
            g_loss = criterion(discriminator(fake_videos.view(batch_size, num_frames, -1)), real_labels)
            g_loss.backward()
            optimizerG.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # Validation step at the end of each epoch
        val_loss = validate_model(generator, discriminator, val_dataloader, criterion, window_size, device)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

    # Plot training and validation losses
    epoch_range = range(1, num_epochs + 1)
    plot_losses_with_validation(d_losses, g_losses, val_losses, epoch_range)

def validate_model(generator, discriminator, dataloader, criterion, window_size, device):
    generator.eval()
    discriminator.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for videos, _ in dataloader:
            videos = videos.to(device)
            for window in sliding_window_sequences(videos, window_size):
                window = torch.stack(window)
                real_labels = torch.ones(window.size(0), 1).to(device)
                validation_loss += criterion(discriminator(window), real_labels).item()
    average_validation_loss = validation_loss / len(dataloader)
    generator.train()
    discriminator.train()
    return average_validation_loss

def plot_losses_with_validation(d_losses, g_losses, val_losses, epoch_range):
    plt.plot(epoch_range, d_losses, label='Discriminator Loss')
    plt.plot(epoch_range, g_losses, label='Generator Loss')
    plt.plot(epoch_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig("losses_validation.png")

def save_model(model, model_name):
    torch.save(model.state_dict(), f'{main_path}saved_models/{model_name}.pth')

def load_model(model, model_name):
    model.load_state_dict(torch.load(f'{main_path}saved_models/{model_name}.pth'))
    model.eval()

def generate_fake_videos(generator, num_samples, latent_dim, num_frames, text_embedding_dim):
    load_model(generator, "generator")
    noise = torch.randn(num_samples, num_frames, latent_dim).to(device)
    text_indices = torch.randint(0, 2000, (num_samples, num_frames)).to(device)
    with torch.no_grad():
        fake_videos = generator(noise, text_indices)
    return fake_videos

def main():
    ids, labels = get_vids(os.path.join(main_path, 'images'))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = create_dataset(ids, labels, transform)
    # Splitting dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True, num_workers=4)

    num_frames = 16  # Define the number of frames
    latent_dim = 100  # Define the latent dimension
    window_size = 5   # Define the window size for validation sliding windows
    num_embeddings = 2000  # Set the number of embeddings for text data
    num_epochs = 20  # Set the number of training epochs

    train(generator, discriminator, train_dataloader, val_dataloader, optimizerG, optimizerD, criterion, num_embeddings, num_epochs, num_frames, latent_dim, window_size, device)

    # Save models after training
    save_model(generator, "generator")
    save_model(discriminator, "discriminator")

    if not os.path.exists(f'{main_path}saved_models'):
        os.makedirs(f'{main_path}saved_models')

    # Generate and save a sample of fake videos
    fake_videos = generate_fake_videos(generator, num_samples=5, latent_dim=latent_dim, num_frames=num_frames, text_embedding_dim=20)
    save_videos(fake_videos, 'generated_videos')

def save_videos(videos, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for i, video in enumerate(videos):
        for j, frame in enumerate(video):
            save_path = f'{folder_name}/video_{i}_frame_{j}.jpg'
            torch.utils.save_image(frame.cpu(), save_path)

if __name__ == "__main__":
    main()
