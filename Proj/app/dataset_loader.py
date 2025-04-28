import torch
from torch.utils.data import Dataset, DataLoader

class MalwareDataset(Dataset):
    def __init__(self, entropy_data, mask_data, image_data, labels):
        self.entropy_data = entropy_data
        self.mask_data = mask_data
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.entropy_data[idx], 
            self.mask_data[idx], 
            self.image_data[idx], 
            self.labels[idx]
        )

def get_loaders(batch_size=128):
    # Load the processed tensors
    entropy_x = torch.load("Proj/Multimodal_Data/entropy_x.pt")
    mask = torch.load("Proj/Multimodal_Data/mask.pt")
    img_x = torch.load("Proj/Multimodal_Data/img_x.pt")
    labels = torch.load("Proj/Multimodal_Data/labels.pt")

    dataset = MalwareDataset(entropy_x, mask, img_x, labels)

    # Split into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
