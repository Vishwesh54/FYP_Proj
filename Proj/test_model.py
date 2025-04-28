import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from app.crossmodal import CrossmodalNet
from app.make_dataset import BuildDataLoader

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def test_model(model, test_loader):
    model.eval()
    model.to(device)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            entropy_data, mask, img_data, labels = batch
            entropy_data = entropy_data.to(device)
            mask = mask.to(device)
            img_data = img_data.to(device)
            labels = labels.to(device)
            outputs, _ = model(entropy_data, img_data, mask)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    target_names = ['Ramnit', 'Lollipop', 'Kelihos ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos ver1', 'Obfuscator.ACY', 'Gatak']
    unique_labels = np.unique(y_true)
    target_names_used = [target_names[i] for i in unique_labels]
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names_used))

def main():
    test_fileListPath = "Microsoft_dataset/malware-classification/trainLabels.csv"
    test_img_folder = "ProcessedData/microsoft_images/_test"
    test_entropy_folder = "ProcessedData/microsoft_entropy_csv/test"
    loader = BuildDataLoader(test_entropy_folder, test_img_folder, test_fileListPath)
    model = CrossmodalNet()
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    test_model(model, loader.val_dataloader())

if __name__ == '__main__':
    main()
