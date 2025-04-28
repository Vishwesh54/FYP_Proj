import torch
import numpy as np
from sklearn.metrics import classification_report

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class trainer():
    def __init__(self, model, trn_loader, val_loader):
        self.model = model
        self.trn_loader = trn_loader
        self.val_loader = val_loader

    def train(self, num_epochs, criterion, optimizer):
        trn_loss_list = []
        val_loss_list = []
        best_val_loss = float('inf')

        self.model.train()
        self.model.to(device)
        for epoch in range(num_epochs):
            trn_loss = 0.0
            for i, (unpack_e, mask, unpack_i, y) in enumerate(self.trn_loader):
                x = unpack_e.to(device)
                x_img = unpack_i.to(device)
                label = y.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()
                model_output, att_score = self.model(x, x_img, mask)
                loss = criterion(model_output, label)
                loss.backward()
                optimizer.step()

                trn_loss += loss.item()
                del loss, model_output

                if (i + 1) % len(self.trn_loader) == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        corr_num = 0
                        total_num = 0
                        for val_e, val_mask, val_i, val_y in self.val_loader:
                            val_x = val_e.to(device)
                            val_img = val_i.to(device)
                            val_mask = val_mask.to(device)
                            val_label = val_y.to(device)

                            val_output, _ = self.model(val_x, val_img, val_mask)
                            v_loss = criterion(val_output, val_label)
                            val_loss += v_loss.item()
                            pred = val_output.argmax(dim=1)
                            corr_num += (pred == val_label).sum().item()
                            total_num += val_label.size(0)
                        acc = corr_num / total_num * 100
                        avg_trn_loss = trn_loss / len(self.trn_loader)
                        avg_val_loss = val_loss / len(self.val_loader)
                        print(f"epoch: {epoch+1}/{num_epochs} | trn loss: {avg_trn_loss:.4f} | val loss: {avg_val_loss:.4f} | acc: {acc:.2f}%")
                        trn_loss_list.append(avg_trn_loss)
                        val_loss_list.append(avg_val_loss)
                        trn_loss = 0.0

                        # Save best model based on validation loss
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save(self.model.state_dict(), "best_model.pt")
                            print(f"Saved best model at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

                    self.model.train()
        # Save final model
        torch.save(self.model.state_dict(), "crossmodal_model_final.pt")
        print("Model saved to crossmodal_model_final.pt")
        self.validate()

    def validate(self):
        y_true = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for val_e, val_mask, val_i, val_y in self.val_loader:
                val_x = val_e.to(device)
                val_img = val_i.to(device)
                val_mask = val_mask.to(device)
                val_label = val_y.to(device)
                val_output, _ = self.model(val_x, val_img, val_mask)
                pred = val_output.argmax(dim=1)
                y_true += val_label.cpu().numpy().tolist()
                y_pred += pred.cpu().numpy().tolist()
        acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
        print(f"Final Validation Accuracy: {acc:.2f}%")
        target_names = [
            'Ramnit', 'Lollipop', 'Kelihos ver3',
            'Vundo', 'Simda', 'Tracur',
            'Kelihos ver1', 'Obfuscator.ACY', 'Gatak'
        ]
        unique_labels = np.unique(y_true)
        target_names_used = [target_names[i] for i in unique_labels]
        print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names_used))
