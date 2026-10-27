import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """ MLP with three hidden layers (128-64-32) used in all reported experiments.
    The input dimension is set from the preprocessed CIC-IDS-2017 features.
    Uses BatchNorm for stable SGD training and Dropout for regularization.
    """
    def __init__(self, input_dim=122, num_classes=2):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# =============================================================================
# CNN-LSTM Model (not used in the reported experiments)
# =============================================================================
class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for label-encoded NSL-KDD data (41 features).
    
    Retained for completeness; not used in the paper's experiments.
    """
    def __init__(self, input_dim=41, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        x = x[:, -1, :] 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    # Test MLP model
    mlp = MLPModel(input_dim=122)
    dummy_mlp = torch.randn(16, 122)
    out_mlp = mlp(dummy_mlp)
    print(f"[MLP] Input: {dummy_mlp.shape} -> Output: {out_mlp.shape}")

    # Test CNN-LSTM model
    cnn_lstm = CNNLSTMModel(input_dim=41)
    dummy_cnn = torch.randn(16, 41)
    out_cnn = cnn_lstm(dummy_cnn)
    print(f"[CNN-LSTM] Input: {dummy_cnn.shape} -> Output: {out_cnn.shape}")

    print("Both models initialized successfully.")
