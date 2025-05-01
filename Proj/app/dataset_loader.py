import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MalwareDataset(Dataset):
    def __init__(self, entropy_data, mask_data, image_data, byte_hist_data, labels):
        self.entropy_data = entropy_data
        self.mask_data = mask_data
        self.image_data = image_data
        self.byte_hist_data = byte_hist_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.entropy_data[idx], 
            self.mask_data[idx], 
            self.image_data[idx],
            self.byte_hist_data[idx],
            self.labels[idx]
        )

def get_loaders(batch_size=128, data_dir='Proj/Multimodal_Data'):
    # Load the processed tensors
    entropy_x    = torch.load(f"{data_dir}/entropy_x.pt")
    mask         = torch.load(f"{data_dir}/mask.pt")
    img_x        = torch.load(f"{data_dir}/img_x.pt")
    byte_hist_x  = torch.load(f"{data_dir}/byte_hist_x.pt")  # new byte histograms
    labels       = torch.load(f"{data_dir}/labels.pt")

    # Ensure all modalities have same length
    assert entropy_x.shape[0] == mask.shape[0] == img_x.shape[0] == byte_hist_x.shape[0] == labels.shape[0], \
        "Modality tensor count mismatch"

    dataset = MalwareDataset(entropy_x, mask, img_x, byte_hist_x, labels)

    # Split into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Example usage:
# train_loader, val_loader = get_loaders(batch_size=32, data_dir='ProcessedData')
