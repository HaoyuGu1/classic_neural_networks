import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2d512(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN2d512, self).__init__()
        
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
        self.fc1 = nn.Linear(1, 512)  # Placeholder value, will be reset in forward()
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(256, 1)

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
                self.fc1 = nn.Linear(self.num_flat_features, 512).to(x.device)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    

class CNN2d128(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN2d128, self).__init__()
        
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
        self.fc1 = nn.Linear(1, 128)  # Placeholder value, will be reset in forward()
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(64, 1)

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
                self.fc1 = nn.Linear(self.num_flat_features, 128).to(x.device)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

class CNN2d64(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN2d64, self).__init__()
        
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
        self.fc1 = nn.Linear(1, 64)  # Placeholder value, will be reset in forward()
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(32, 1)

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
                self.fc1 = nn.Linear(self.num_flat_features, 64).to(x.device)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


class CNN2d1fc(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN2d1fc, self).__init__()
        
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
    

class CNN2dgap(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN2dgap, self).__init__()
        
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
        # self.num_flat_features = None  # This will be dynamically calculated
        
        # Fully connected layers with dropout
        # The 'num_flat_features' will be calculated in the forward pass before using it here
        # self.fc1 = nn.Linear(1, 128)  # Placeholder value, will be reset in forward()
        # self.bn4 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(dropout_prob)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn5 = nn.BatchNorm1d(64)
        # self.dropout2 = nn.Dropout(dropout_prob)
        # self.fc3 = nn.Linear(64, 1)
        # self.fc = None
        self.gap = nn.AdaptiveAvgPool2d(1)  # Reduces spatial dimensions to 1x1
        self.fc = nn.Linear(64, 1)          # Directly map channels to output

    
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

        # self.num_flat_features = x.view(x.size(0), -1).shape[1]
        
        # # Dynamically calculate the number of flat features
        # if self.fc1 is None:
        #     with torch.no_grad():
        #         # self.fc1 = nn.Linear(self.num_flat_features, 128).to(x.device)
        #         # self.fc = nn.Linear(self.num_flat_features, 1).to(x.device)
        #         self.fc1 = nn.Linear(num_flat_features, num_flat_features // 2)  # Halve the features
        #         self.fc2 = nn.Linear(num_flat_features // 2, 1)  

        # x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # # x = self.fc1(x)
        # x = self.fc(x)

        x = self.gap(x)       # Shape: [batch, 64, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)        # Shape: [batch, 1]
        return x