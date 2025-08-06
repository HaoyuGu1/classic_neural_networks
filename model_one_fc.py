import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2d(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN2d, self).__init__()
        
        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Placeholder for the number of features for the first fully connected layer
        self.num_flat_features = None  # This will be dynamically calculated
        
        # Fully connected layers with dropout
        # The 'num_flat_features' will be calculated in the forward pass before using it here
        # self.fc1 = nn.Linear(1, 128)  # Placeholder value, will be reset in forward()
        # self.bn4 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(dropout_prob)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn5 = nn.BatchNorm1d(64)
        # self.dropout2 = nn.Dropout(dropout_prob)
        # self.fc3 = nn.Linear(64, 1)
        self.fc = None
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Dynamically calculate the number of flat features
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                # self.fc1 = nn.Linear(self.num_flat_features, 128).to(x.device)
                self.fc = nn.Linear(self.num_flat_features, 1).to(x.device)
       
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # x = self.fc1(x)
        x = self.fc(x)
        return x