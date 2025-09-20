import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

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
    

class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
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


class RegressionViT(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.5, image_size=448, num_channels=3):  # Adjust image_size to 448, assume RGB images
        super(RegressionViT, self).__init__()

        self.config = ViTConfig(
            image_size=image_size,  # Updated image size to 448
            num_channels=num_channels,  # Assuming RGB images
            patch_size=32,  # Keeping the patch size as 16x16
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
        )

        self.vit = ViTModel(self.config)

        self.preprocess = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.config.num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.num_channels),
            nn.ReLU(),
        )

        # Adjust the size of the input layer of the regressor to match the output dimension of the ViT.
        # This requires calculating the number of output features from ViT, which depends on its configuration.
        # Assuming the dimensionality does not change for simplicity, but this may need to be adjusted based on the ViTModel implementation.
        
        # Define the new DNN architecture for the regressor
        self.fc1 = nn.Linear(self.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)  # Output layer adjusted for num_classes
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.preprocess(x)
        outputs = self.vit(pixel_values=x)
        x = outputs.last_hidden_state[:, 0]  # Use the representation of the [CLS] token

        # Pass through the new DNN
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)  # No activation for the final layer in regression

        return x    