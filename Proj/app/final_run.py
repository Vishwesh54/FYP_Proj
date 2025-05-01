import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from make_dataset import MultimodalDataset  # returns entropy, mask, img, x_bytes, label
from crossmodal import CrossmodalNet          # accepts entropy, img, x_bytes, mask

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size    = 16
num_epochs    = 50
learning_rate = 0.001
num_folds     = 10

# File paths
fileListPath = "Microsoft_dataset/malware-classification/trainLabels.csv"
fn_img       = "ProcessedData/microsoft_images/_train"
fn_entropy   = "ProcessedData/microsoft_entropy_csv/train"
fn_bytes     = "ProcessedData/byte_histograms"   # byte-histogram folder

# Load full dataset (now includes byte histograms)
dataset = MultimodalDataset(fileListPath, fn_img, fn_entropy, fn_bytes)

# K-Fold setup
num_samples = len(dataset)
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(num_samples)), 1):
    print(f"\n--- Fold {fold} ---")

    # create subsets & loaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)

    # model, loss, optimizer
    model     = CrossmodalNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for entropy, mask, img, x_bytes, labels in train_loader:
            entropy, mask, img, x_bytes, labels = (
                entropy.to(device),
                mask.to(device),
                img.to(device),
                x_bytes.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(entropy, img, x_bytes, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for entropy, mask, img, x_bytes, labels in val_loader:
            entropy, mask, img, x_bytes, labels = (
                entropy.to(device),
                mask.to(device),
                img.to(device),
                x_bytes.to(device),
                labels.to(device),
            )

            outputs = model(entropy, img, x_bytes, mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    print(f"Fold {fold} Accuracy: {acc:.2f}%")
    print(classification_report(y_true, y_pred, digits=4))
    fold_accuracies.append(acc)

# Final result
print(f"\nAverage Accuracy across {num_folds} folds: {np.mean(fold_accuracies):.2f}%")
