import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import os
from skimage import io
import pandas as pd
from collections import Counter

def load_img_data(fileListPath, folder):
    transform = transforms.Compose([transforms.ToTensor()])
    df = pd.read_csv(fileListPath, sep=',')
    malware_name_label = df.values
    dirTarget = os.path.join(os.getcwd(), folder)

    filesLen = len(malware_name_label)
    data_with_padding = np.zeros((filesLen, 64, 784))
    y_label_number = np.zeros(filesLen)
    valid_filenames = []
    index = 0

    for entryIndex in tqdm(range(filesLen), desc="Loading images"):
        fetched_name_label = malware_name_label[entryIndex]
        filename = fetched_name_label[0]
        name_with_extension = filename + '.png'
        path = os.path.join(dirTarget, name_with_extension)

        try:
            data = io.imread(path, as_gray=True)
            data = transform(data).reshape(64, -1).numpy()
            data_with_padding[index] = data
            y_label_number[index] = fetched_name_label[1] - 1
            valid_filenames.append(filename)
            index += 1
        except FileNotFoundError:
            print(f"Image missing: {name_with_extension}")

    return data_with_padding[:index], y_label_number[:index], valid_filenames

def load_pefile_data(fileListPath, folder, w=14, MaxChunkLen=3600):
    df = pd.read_csv(fileListPath, sep=',')
    malware_name_label = df.values
    dirTarget = os.path.join(os.getcwd(), folder)

    filesLen = len(malware_name_label)
    data_with_padding = np.zeros((filesLen, MaxChunkLen, w))
    data_with_mask = np.full((filesLen, MaxChunkLen, 1), 1)
    y_label_number = np.zeros(filesLen)
    valid_filenames = []
    index = 0

    for entryIndex in tqdm(range(filesLen), desc="Loading entropy"):
        fetched_name_label = malware_name_label[entryIndex]
        filename = fetched_name_label[0]
        name_with_extension = filename + '.csv'
        path = os.path.join(dirTarget, name_with_extension)

        try:
            df_haar = pd.read_csv(path, sep=',', header=None)
            data = df_haar.values[:, :w]

            if len(data) < MaxChunkLen:
                pad = MaxChunkLen - len(data)
                data_mask = np.vstack((np.ones((len(data), 1)), np.zeros((pad, 1))))
                data = np.vstack((data, np.zeros((pad, w))))
            else:
                data = data[:MaxChunkLen]
                data_mask = np.ones((MaxChunkLen, 1))

            data_with_padding[index] = data
            data_with_mask[index] = data_mask
            y_label_number[index] = fetched_name_label[1] - 1
            valid_filenames.append(filename)
            index += 1
        except FileNotFoundError:
            print(f"Entropy file missing: {name_with_extension}")

    data_with_padding = np.transpose(data_with_padding[:index], (0, 2, 1))
    return data_with_padding, y_label_number[:index], data_with_mask[:index], valid_filenames

class MultimodalDataset(Dataset):
    def __init__(self, fileListPath, fn_img, fn_entropy):
        img_data, img_y, img_files = load_img_data(fileListPath, fn_img)
        entropy_data, entropy_y, entropy_mask, entropy_files = load_pefile_data(fileListPath, fn_entropy)

        common_files = sorted(set(img_files) & set(entropy_files))
        print(f"Found {len(common_files)} common samples.")

        img_map = {name: i for i, name in enumerate(img_files)}
        ent_map = {name: i for i, name in enumerate(entropy_files)}

        self.img_data = []
        self.entropy_data = []
        self.entropy_mask = []
        self.labels = []

        for fname in common_files:
            img_idx = img_map[fname]
            ent_idx = ent_map[fname]

            if img_y[img_idx] != entropy_y[ent_idx]:
                print(f"Label mismatch for {fname}, skipping.")
                continue

            self.img_data.append(img_data[img_idx])
            self.entropy_data.append(entropy_data[ent_idx])
            self.entropy_mask.append(entropy_mask[ent_idx])
            self.labels.append(img_y[img_idx])

        self.img_data = np.array(self.img_data)
        self.entropy_data = np.array(self.entropy_data)
        self.entropy_mask = np.array(self.entropy_mask)
        self.labels = np.array(self.labels)

        print("Class distribution:", Counter(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.entropy_data[idx], dtype=torch.float32),
            torch.tensor(self.entropy_mask[idx], dtype=torch.float32),
            torch.tensor(self.img_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.int64)
        )
