import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import os
from skimage import io
import pandas as pd
from collections import Counter

##########################################
# Helper functions for loading data
##########################################

def load_img_data(fileListPath, folder):
    """Loads images and labels, and also returns the filenames loaded."""
    transform = transforms.Compose([transforms.ToTensor()])
    df = pd.read_csv(fileListPath, sep=',')
    malware_name_label = df.values
    dirTarget = os.path.join(os.getcwd(), folder)
    filesLen = len(malware_name_label)
    data_with_padding = np.zeros((filesLen, 64, 784))
    y_label_number = np.zeros(filesLen)
    valid_filenames = []  # Collect filenames for which images were loaded
    index = 0

    for entryIndex in tqdm(range(filesLen), desc="Loading images"):
        fetched_name_label = malware_name_label[entryIndex]
        # Use the filename without extension as the identifier
        filename = fetched_name_label[0]
        name_with_extension = filename + '.png'
        pathTarget = os.path.join(dirTarget, name_with_extension)
        try:
            data_non_pad = io.imread(pathTarget, as_gray=True)
            # Convert to tensor then to NumPy array and reshape
            data_non_pad = transform(data_non_pad).reshape(64, -1).numpy()
            data_with_padding[index] = data_non_pad
            y_label_number[index] = fetched_name_label[1] - 1
            valid_filenames.append(filename)
            index += 1
        except FileNotFoundError:
            print(f"File does not exist: {name_with_extension}")
    
    data_with_padding = data_with_padding[:index]
    y_label_number = y_label_number[:index]
    return data_with_padding, y_label_number, valid_filenames

def load_pefile_data(fileListPath, folder, w=14, MaxChunkLen=3600):
    """Loads entropy data and labels, and returns loaded filenames."""
    df = pd.read_csv(fileListPath, sep=',')
    malware_name_label = df.values
    dirTarget = os.path.join(os.getcwd(), folder)
    filesLen = len(malware_name_label)
    data_with_padding = np.zeros((filesLen, MaxChunkLen, w))
    data_with_mask = np.full((filesLen, MaxChunkLen, 1), 1)
    y_label_number = np.zeros(filesLen)
    valid_filenames = []
    index = 0

    for entryIndex in tqdm(range(filesLen), desc="Loading entropy data"):
        fetched_name_label = malware_name_label[entryIndex]
        filename = fetched_name_label[0]
        name_with_extension = filename + '.csv'
        pathTarget = os.path.join(dirTarget, name_with_extension)
        try:
            df_haar = pd.read_csv(pathTarget, sep=',', header=None, index_col=None)
            data_non_pad = df_haar.values[:, :w]
            # Pad data if needed
            if len(data_non_pad) < MaxChunkLen:
                tp = MaxChunkLen - len(data_non_pad)
                padArray = np.zeros((tp, w))
                padArray_mask = np.zeros((tp, 1))
                data_mask = np.ones((len(data_non_pad), 1))
                data_non_pad = np.vstack((data_non_pad, padArray))
                data_mask = np.vstack((data_mask, padArray_mask))
            else:
                data_non_pad = data_non_pad[:MaxChunkLen]
                data_mask = np.ones((MaxChunkLen, 1))
            data_with_padding[index] = data_non_pad
            data_with_mask[index] = data_mask
            y_label_number[index] = fetched_name_label[1] - 1
            valid_filenames.append(filename)
            index += 1
        except FileNotFoundError:
            print(f"File does not exist: {name_with_extension}")
    
    data_with_padding = data_with_padding[:index]
    y_label_number = y_label_number[:index]
    data_with_mask = data_with_mask[:index]
    
    # Transpose entropy data so that shape becomes [N, w, MaxChunkLen]
    data_with_padding = np.transpose(data_with_padding, (0, 2, 1))
    return data_with_padding, y_label_number, data_with_mask, valid_filenames

##########################################
# Dataset and Data Module definitions
##########################################

class MultimodalDataset(Dataset):
    def __init__(self, fileListPath, fn_img, fn_entropy):
        # Load both modalities along with their valid filenames
        img_data, img_y, valid_filenames_img = load_img_data(fileListPath, fn_img)
        entropy_data, entropy_y, entropy_mask, valid_filenames_entropy = load_pefile_data(fileListPath,fn_entropy)
        
        # Get the intersection of filenames loaded in both modalities
        valid_set = set(valid_filenames_img) & set(valid_filenames_entropy)
        if not valid_set:
            raise RuntimeError("No common filenames were found between image and entropy datasets!")
        
        # Create dictionaries for fast lookup: filename -> (data, label)
        img_dict = {fname: (img_data[i], img_y[i]) for i, fname in enumerate(valid_filenames_img) if fname in valid_set}
        entropy_dict = {fname: (entropy_data[i], entropy_y[i], entropy_mask[i]) for i, fname in enumerate(valid_filenames_entropy) if fname in valid_set}
        
        # Now build aligned lists where both modalities are guaranteed to have the same files, sorted or in arbitrary order.
        common_filenames = sorted(list(valid_set))
        self.img_data = []
        self.img_y = []
        self.entropy_data = []
        self.entropy_y = []
        self.entropy_mask = []
        
        for fname in common_filenames:
            img_sample, label_img = img_dict[fname]
            entropy_sample, label_entropy, mask_sample = entropy_dict[fname]
            # Sanity check: labels should match
            if label_img != label_entropy:
                print(f"Label mismatch for file {fname}: image label {label_img} vs entropy label {label_entropy}")
                continue  # Skip mismatches
            self.img_data.append(img_sample)
            self.img_y.append(label_img)
            self.entropy_data.append(entropy_sample)
            self.entropy_y.append(label_entropy)
            self.entropy_mask.append(mask_sample)
        
        # Convert lists to numpy arrays for later tensor conversion
        self.img_data = np.array(self.img_data)
        self.img_y = np.array(self.img_y)
        self.entropy_data = np.array(self.entropy_data)
        self.entropy_y = np.array(self.entropy_y)
        self.entropy_mask = np.array(self.entropy_mask)
        
        print("Aligned Image Data Class Distribution:", Counter(self.img_y))
        print("Aligned Entropy Data Class Distribution:", Counter(self.entropy_y))

    def __len__(self):
        return len(self.img_y)

    def __getitem__(self, idx):
        # Return the sample as tensors
        entropy_x = torch.tensor(self.entropy_data[idx], dtype=torch.float32)
        mask = torch.tensor(self.entropy_mask[idx], dtype=torch.float32)
        img_x = torch.tensor(self.img_data[idx], dtype=torch.float32)
        label = torch.tensor(self.img_y[idx], dtype=torch.int64)
        return [entropy_x, mask, img_x, label]

class MultimodalDataModule():
    def __init__(self, fileListPath, fn_img, fn_entropy, batch_size=128):
        self.batch_size = batch_size
        self.all_train_dataset = MultimodalDataset(fileListPath, fn_img, fn_entropy)
        N = len(self.all_train_dataset)
        tr = int(N * 0.82)
        va = N - tr
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
            self.all_train_dataset, [tr, va], generator=torch.Generator().manual_seed(91)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

class BuildDataLoader():
    def __init__(self, train_fn_entropy=None, train_fn_img=None, fileListPath=None):
        MD = MultimodalDataModule(fileListPath, train_fn_img, train_fn_entropy)
        self.trn_loader = MD.train_dataloader()
        self.val_loader = MD.val_dataloader()

##########################################
# Usage example: saving preprocessed data
##########################################

if __name__ == '__main__':
    # Define file paths and folders (update these paths as needed)
    fileListPath = "Microsoft_dataset/malware-classification/trainLabels.csv"
    # fileListPathEntropy = "Microsoft_dataset/malware-classification/trainLabels.csv"
    img_folder = "ProcessedData/microsoft_images/_test"
    entropy_folder = "ProcessedData/microsoft_entropy_csv/test"
    
    # Build data loader; note the order: (fn_img, fn_entropy) refers to the folder names.
    loader = BuildDataLoader(entropy_folder, img_folder, fileListPath)
    
    # Example: accumulate training data and save to .pt files
    all_entropy = []
    all_mask = []
    all_img = []
    all_labels = []
    
    for batch in loader.trn_loader:
        entropy_x, mask, img_x, labels = batch
        all_entropy.append(entropy_x)
        all_mask.append(mask)
        all_img.append(img_x)
        all_labels.append(labels)
    
    all_entropy = torch.cat(all_entropy, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    all_img = torch.cat(all_img, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    multimodal_path = "Proj/Multimodal_Data_Test/"
    os.makedirs(multimodal_path, exist_ok=True)
    
    torch.save(all_entropy, os.path.join(multimodal_path, "entropy_x.pt"))
    torch.save(all_mask, os.path.join(multimodal_path, "mask.pt"))
    torch.save(all_img, os.path.join(multimodal_path, "img_x.pt"))
    torch.save(all_labels, os.path.join(multimodal_path, "labels.pt"))
    
    print("Saved preprocessed training data to .pt files in 'Proj/Multimodal_Data_Test/'")
    
    # Inspect one batch from the training loader to verify shapes
    for batch in loader.trn_loader:
        entropy_x, mask, img_x, labels = batch
        print("Entropy Data Shape:", entropy_x.shape)
        print("Mask Shape:", mask.shape)
        print("Image Data Shape:", img_x.shape)
        print("Labels Shape:", labels.shape)
        break
