import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel


# model 1
class CNN_3_3_16_512(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN_3_3_16_512, self).__init__()
        
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
    
# model 2
class CNN_2_2_16_512(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(CNN_2_2_16_512, self).__init__()
        
        # First convolutional block - REDUCED from 32 to 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional block - REDUCED from 64 to 32  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Placeholder for the number of features
        self.num_flat_features = None
        
        # First fully connected layer
        self.fc1 = nn.Linear(1, 512)  # Placeholder
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Forward pass remains the same...
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                self.fc1 = nn.Linear(self.num_flat_features, 512).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    






    
class CNN_3_3_16_256(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN_3_3_16_256, self).__init__()
        
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
        self.fc1 = nn.Linear(1, 256)  # Placeholder value, will be reset in forward()
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(128, 1)

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
                self.fc1 = nn.Linear(self.num_flat_features, 256).to(x.device)
        
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
    
class CNN_2_2_16_256(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(CNN_2_2_16_256, self).__init__()
        
        # First convolutional block - REDUCED from 32 to 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional block - REDUCED from 64 to 32  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Placeholder for the number of features
        self.num_flat_features = None
        
        # First fully connected layer
        self.fc1 = nn.Linear(1, 256)  # Placeholder
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Forward pass remains the same...
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                self.fc1 = nn.Linear(self.num_flat_features, 256).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    



class CNN_3_3_8_512(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN_3_3_8_512, self).__init__()
        
        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)
        
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
    

class CNN_2_2_8_512(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(CNN_2_2_8_512, self).__init__()
        
        # First convolutional block - REDUCED from 32 to 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolutional block - REDUCED from 64 to 32  
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Placeholder for the number of features
        self.num_flat_features = None
        
        # First fully connected layer
        self.fc1 = nn.Linear(1, 512)  # Placeholder
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Forward pass remains the same...
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                self.fc1 = nn.Linear(self.num_flat_features, 512).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    




class CNN_3_3_8_256(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN_3_3_8_256, self).__init__()
        
        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Placeholder for the number of features for the first fully connected layer
        self.num_flat_features = None  # This will be dynamically calculated
        
        # Fully connected layers with dropout
        # The 'num_flat_features' will be calculated in the forward pass before using it here
        self.fc1 = nn.Linear(1, 256)  # Placeholder value, will be reset in forward()
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(128, 1)

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
                self.fc1 = nn.Linear(self.num_flat_features, 256).to(x.device)
        
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
    

class CNN_2_2_8_256(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(CNN_2_2_8_256, self).__init__()
        
        # First convolutional block - REDUCED from 32 to 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolutional block - REDUCED from 64 to 32  
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Placeholder for the number of features
        self.num_flat_features = None
        
        # First fully connected layer
        self.fc1 = nn.Linear(1, 256)  # Placeholder
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Forward pass remains the same...
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                self.fc1 = nn.Linear(self.num_flat_features, 256).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x    
    


    
class CNN_3_3_8_128(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN_3_3_8_128, self).__init__()
        
        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)
        
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
    

class CNN_2_2_8_128(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(CNN_2_2_8_128, self).__init__()
        
        # First convolutional block - REDUCED from 32 to 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolutional block - REDUCED from 64 to 32  
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Placeholder for the number of features
        self.num_flat_features = None
        
        # First fully connected layer
        self.fc1 = nn.Linear(1, 128)  # Placeholder
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Forward pass remains the same...
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                self.fc1 = nn.Linear(self.num_flat_features, 128).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x    
    



    
    
class CNN_3_3_16_128(nn.Module):
    def __init__(self, dropout_prob = 0.25):
        super(CNN_3_3_16_128, self).__init__()
        
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
    

class CNN_2_2_16_128(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(CNN_2_2_16_128, self).__init__()
        
        # First convolutional block - REDUCED from 32 to 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional block - REDUCED from 64 to 32  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Placeholder for the number of features
        self.num_flat_features = None
        
        # First fully connected layer
        self.fc1 = nn.Linear(1, 128)  # Placeholder
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Forward pass remains the same...
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                self.fc1 = nn.Linear(self.num_flat_features, 128).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x    