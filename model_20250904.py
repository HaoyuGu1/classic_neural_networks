import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel


class DNN_5_512(nn.Module):
    def __init__(self, input_size):
        super(DNN_5_512, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization after first layer
        self.bn2 = nn.BatchNorm1d(256)  # Batch normalization after second layer
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization after third layer
        self.bn4 = nn.BatchNorm1d(64)   # Batch normalization after fourth layer

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x


class DNN_5_256(nn.Module):
    def __init__(self, input_size):
        super(DNN_5_256, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization after first layer
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization after second layer
        self.bn3 = nn.BatchNorm1d(64)  # Batch normalization after third layer
        self.bn4 = nn.BatchNorm1d(32)   # Batch normalization after fourth layer

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x



class DNN_5_128(nn.Module):
    def __init__(self, input_size):
        super(DNN_5_128, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization after first layer
        self.bn2 = nn.BatchNorm1d(64)  # Batch normalization after second layer
        self.bn3 = nn.BatchNorm1d(32)  # Batch normalization after third layer
        self.bn4 = nn.BatchNorm1d(16)   # Batch normalization after fourth layer

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x