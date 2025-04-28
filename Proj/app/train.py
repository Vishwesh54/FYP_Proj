import torch
import torch.nn as nn
import torch.optim as optim
from crossmodal import CrossmodalNet
from trainer import trainer
from dataset_loader import get_loaders  # You must have this implemented

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_loaders(batch_size=16)
model = CrossmodalNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer_instance = trainer(model, train_loader, val_loader)
trainer_instance.train(num_epochs=15, criterion=criterion, optimizer=optimizer)