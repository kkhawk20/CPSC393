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
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn as nn

main_path = "/app/rundir/CPSC393/FinalProject/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
Utility Functions
"""
# Function to get the video ids from the json file

# Loading of the data into a dataframe
def loading_data():
    def get_video_ids(json_list):
        video_ids = []
        for item in json_list:
            video_id = item.get('video_id', None)
            if video_id and os.path.exists(f"{main_path}/videos_raw/{video_id}.mp4"):
                video_ids.append(video_id)
        return video_ids

    wlasl_df = pd.read_json(main_path + "WLASL_v0.3.json")
    print(wlasl_df.head())

    with open(main_path+'WLASL_v0.3.json', 'r') as data_file:
        json_data = data_file.read()
    instance_json = json.loads(json_data)

    # Creating a dataframe to store the features from the key file
    features_df = pd.DataFrame(columns=['gloss', 'video_id'])
    for row in wlasl_df.iterrows():
        ids = get_video_ids(row[1][1])
        word = [row[1][0]] * len(ids)
        df = pd.DataFrame(list(zip(word, ids)), columns=features_df.columns)
        features_df = pd.concat([features_df, df], ignore_index=True)

    return features_df

# Creating a function that gets the videos from the dataset of videos
def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    listOfCats.remove('.DS_Store')
    ids = []
    labels = []
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        # print(path2catg)
        listOfSubCats = os.listdir(path2catg)
        path2subCats= [os.path.join(path2catg,los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([catg]*len(listOfSubCats))
    return ids, labels, listOfCats 

def create_dict():

    # Creating a dictionary to hold all 2000 labels and indexes for references
    sub_folder_jpg = 'images'
    path2ajpgs = sub_folder_jpg

    all_vids, all_labels, all_cats = get_vids(path2ajpgs)

    labels_dict = {}
    ind = 0
    for label in all_cats:
        labels_dict[label] = ind
        ind += 1

    with open('labels_dict.txt', 'w') as file:
        for key in labels_dict.keys():
            file.write(key + ": " + str(labels_dict[key]) + "\n")
    # print("Saved to labels file!")

    num_classes = 2000
    unique_ids = [id_ for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
    unique_labels = [label for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
    # print(len(unique_ids), len(unique_labels))
    return unique_ids, unique_labels, labels_dict

def split_data(unique_ids, unique_labels):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]
    # print(len(train_ids), len(train_labels)) 

    test_ids = [unique_ids[ind] for ind in test_indx]
    test_labels = [unique_labels[ind] for ind in test_indx]
    # print(len(test_ids), len(test_labels))
    # print(train_ids[:5], train_labels[:5])
    return train_ids, train_labels, test_ids, test_labels


'''
DEFINING THE DATASET INITIALIZER AND CREATOR
'''

def dataset_creator(labels_dict, train_ids, train_labels, test_ids, test_labels, timesteps=16):
        
    np.random.seed(2020)
    random.seed(2020)
    torch.manual_seed(2020)

    class VideoDataset(Dataset):
        def __init__(self, ids, labels, transform):      
            self.transform = transform
            self.ids = ids
            self.labels = labels
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, idx):
            path2imgs=glob.glob(self.ids[idx]+"/*.jpg")
            path2imgs = path2imgs[:timesteps]
            label = labels_dict[self.labels[idx]]
            frames = []
            for p2i in path2imgs:
                frame = Image.open(p2i)
                frames.append(frame)
            
            seed = np.random.randint(1e9)        
            frames_tr = []
            for frame in frames:
                random.seed(seed)
                np.random.seed(seed)
                frame = self.transform(frame)
                frames_tr.append(frame)
            if len(frames_tr)>0:
                frames_tr = torch.stack(frames_tr)
            return frames_tr, label

    model_type = "rnn"    
    timesteps =16
    if model_type == "rnn":
        h, w =224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]


    train_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.RandomHorizontalFlip(p=0.5),  
                transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])  

    train_ds = VideoDataset(ids= train_ids, labels= train_labels, transform= train_transformer)
    # print(len(train_ds))
    imgs, label = train_ds[10]
    # print(imgs.shape, label, torch.min(imgs), torch.max(imgs))

    test_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ]) 
    test_ds = VideoDataset(ids= test_ids, labels= test_labels, transform= test_transformer)
    # print(len(test_ds))
    imgs, label = test_ds[5]
    # print(imgs.shape, label, torch.min(imgs), torch.max(imgs))
    return train_ds, test_ds

'''
MODEL DEFINITIONS
'''

def train_model(train_ds, test_ds):
        
    class TextEncoder(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super(TextEncoder, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        def forward(self, text_input):
            embedded = self.embedding(text_input)
            _, (hidden, _) = self.lstm(embedded)
            return hidden[-1]

    class Generator(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(Generator, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)  # Output dim is flattened image size

        def forward(self, text_features, random_noise):
            combined_input = torch.cat((text_features, random_noise), dim=1)
            output, _ = self.lstm(combined_input)
            video_frames = self.fc(output)
            video_frames = video_frames.view(-1, 30, 3, 224, 224)  # reshape to video format
            return video_frames

    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.lstm = nn.LSTM(input_dim, 256, batch_first=True)
            self.fc = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, video_frames):
            video_frames_flat = video_frames.view(video_frames.size(0), 30, -1)  # flatten frames
            _, (hidden, _) = self.lstm(video_frames_flat)
            out = self.fc(hidden[-1])
            return self.sigmoid(out)

    # Assuming each text has a vocabulary size of 1000, embedding dimension of 100, and hidden dimension of 256
    text_encoder = TextEncoder(vocab_size=1000, embedding_dim=100, hidden_dim=256).to(device)

    # Generator and Discriminator
    generator = Generator(input_dim=256, hidden_dim=512, output_dim=3*224*224*30).to(device)  # Adjust dimensions according to your setup
    discriminator = Discriminator(input_dim=3*224*224*30).to(device)  # Assuming flattened video frames

    # Noise dimension for generator input
    noise_dim = 100

    # Optimizers
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Log interval for printing out metrics
    log_interval = 50

    num_epochs = 10

    def my_collate(batch):
        # Separate the frames and labels
        frames, labels = zip(*batch)
        frames_padded = pad_sequence(frames, batch_first=True, padding_value=0)
        labels = torch.tensor(labels)
        return frames_padded, labels

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=my_collate  # Here is where my_collate is used
    )

    for epoch in range(num_epochs):
        for i, (videos, texts) in enumerate(train_loader):
            # videos: real video data
            # texts: corresponding text data

            # Get text features
            text_features = text_encoder(texts)

            # Train Discriminator
            ## Real videos
            real_videos = videos.to(device)
            real_labels = torch.ones(videos.size(0), 1).to(device)
            
            ## Generated videos
            noise = torch.randn(videos.size(0), noise_dim).to(device)
            fake_videos = generator(text_features, noise).detach()  # Detach to avoid training G on these labels
            fake_labels = torch.zeros(videos.size(0), 1).to(device)
            
            # Combine real and fake videos
            all_videos = torch.cat([real_videos, fake_videos], dim=0)
            all_labels = torch.cat([real_labels, fake_labels], dim=0)
            
            # Discriminator output
            discriminator_preds = discriminator(all_videos)
            d_loss = criterion(discriminator_preds, all_labels)

            # Backprop and optimize
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            noise = torch.randn(videos.size(0), noise_dim).to(device)
            fake_videos = generator(text_features, noise)
            fake_labels = torch.ones(videos.size(0), 1).to(device)  # Generator wants discriminator to mistake these as real

            discriminator_preds = discriminator(fake_videos)
            g_loss = criterion(discriminator_preds, fake_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            if i % log_interval == 0:
                print("Epoch [{}/{}], Step [{}/{}], D Loss: {}, G Loss: {}".format(epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item()))


'''
MAIN BLOCK
'''
def main():
    features_df = loading_data()
    unique_ids, unique_labels, labels_dict = create_dict()
    train_ids, train_labels, test_ids, test_labels = split_data(unique_ids, unique_labels)
    train_ds, test_ds = dataset_creator(labels_dict, train_ids, train_labels, test_ids, test_labels)
    print("Dataset created!")

    # train_model(train_ds, test_ds)

main()