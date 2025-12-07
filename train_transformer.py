import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dataset class for tabular data
class CreditDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Simple Transformer Encoder for tabular data
class TabTransformer(nn.Module):
    def __init__(self, n_features, dim_model=64, n_heads=8, n_layers=2, n_classes=2):
        super(TabTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=n_heads)
        self.embed = nn.Linear(n_features, dim_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(dim_model, n_classes)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)  # shape: (batch_size, seq_len=1, dim_model)
        x = self.transformer(x)
        x = x.mean(dim=1)  # pooling
        return self.classifier(x)

def main():
    # Load preprocessed data
    df = pd.read_csv('credit_risk_cleaned.csv')  # change to your chosen cleaned dataset
    # Assuming last column is label or create labels accordingly—here random binary for demo
    np.random.seed(42)
    y = np.random.randint(0, 2, size=len(df))  # Replace with actual target if available
    X = df.values

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = CreditDataset(X_train, y_train)
    val_dataset = CreditDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabTransformer(n_features=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), 'tabtransformer_credit_model.pth')
    print("Model training complete and saved as tabtransformer_credit_model.pth")

if __name__ == '__main__':
    main()
