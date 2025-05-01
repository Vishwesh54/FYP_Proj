import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import os
from skimage import io
import pandas as pd
from collections import Counter

# helper to load byte histograms
import pathlib

def load_byte_histograms(folder, valid_filenames):
    """
    Load precomputed byte histograms (.npy) for each filename in valid_filenames.
    Returns an array of shape (N, 256).
    """
    histograms = []
    for fname in valid_filenames:
        path = os.path.join(folder, f"{fname}.npy")
        try:
            hist = np.load(path)
        except FileNotFoundError:
            # if missing, use zeros
            hist = np.zeros(256, dtype=float)
        histograms.append(hist)
    return np.stack(histograms)


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
        filename, label = malware_name_label[entryIndex]
        name_with_extension = filename + '.png'
        path = os.path.join(dirTarget, name_with_extension)

        try:
            data = io.imread(path, as_gray=True)
            data = transform(data).reshape(64, -1).numpy()
            data_with_padding[index] = data
            y_label_number[index] = label - 1
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
        filename, label = malware_name_label[entryIndex]
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
            y_label_number[index] = label - 1
            valid_filenames.append(filename)
            index += 1
        except FileNotFoundError:
            print(f"Entropy file missing: {name_with_extension}")

    data_with_padding = np.transpose(data_with_padding[:index], (0, 2, 1))
    return data_with_padding, y_label_number[:index], data_with_mask[:index], valid_filenames


class MultimodalDataset(Dataset):
    def __init__(self, fileListPath, fn_img, fn_entropy, fn_bytes):
        # load each modality
        img_data, img_y, img_files = load_img_data(fileListPath, fn_img)
        entropy_data, entropy_y, entropy_mask, entropy_files = load_pefile_data(fileListPath, fn_entropy)

        # find common samples
        common_files = sorted(set(img_files) & set(entropy_files))
        print(f"Found {len(common_files)} common img+entropy samples.")

        # filter to common
        img_map = {name: i for i, name in enumerate(img_files)}
        ent_map = {name: i for i, name in enumerate(entropy_files)}

        self.img_data = []
        self.entropy_data = []
        self.entropy_mask = []
        self.labels = []
        self.files = []

        for fname in common_files:
            i = img_map[fname]
            e = ent_map[fname]
            if img_y[i] != entropy_y[e]:
                print(f"Label mismatch for {fname}, skipping.")
                continue
            self.img_data.append(img_data[i])
            self.entropy_data.append(entropy_data[e])
            self.entropy_mask.append(entropy_mask[e])
            self.labels.append(img_y[i])
            self.files.append(fname)

        # convert lists to arrays
        self.img_data = np.array(self.img_data)
        self.entropy_data = np.array(self.entropy_data)
        self.entropy_mask = np.array(self.entropy_mask)
        self.labels = np.array(self.labels)

        # load byte histograms for these files
        self.byte_hist = load_byte_histograms(fn_bytes, self.files)
        print("Class distribution:", Counter(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return entropy, mask, image, byte_hist, label
        return (
            torch.tensor(self.entropy_data[idx], dtype=torch.float32),
            torch.tensor(self.entropy_mask[idx], dtype=torch.float32),
            torch.tensor(self.img_data[idx], dtype=torch.float32),
            torch.tensor(self.byte_hist[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.int64)
        )

# Example DataLoader creation:
# ds = MultimodalDataset('trainLabels.csv', 'images', 'entropy', 'ProcessedData/byte_histograms')
# loader = DataLoader(ds, batch_size=32, shuffle=True)
