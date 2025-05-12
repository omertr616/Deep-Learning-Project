import torch
import torch.nn as nn

class ClassifierCifar(nn.Module):
    def __init__(self, features_size=128 ,dropout = 0.1):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(features_size, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
    
        #     nn.Linear(64, 10),
        # )
        
        self.cnn = nn.Sequential(
            nn.Unflatten(1, (2, 8, 8)),
            nn.Conv2d(2, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            # (4, 4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            # (2, 2)
            nn.Flatten(),
            nn.Linear(2*2*512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )
        
        # self.cnn = nn.Sequential(
        #     nn.Unflatten(1, (2, 8, 8)),
            
        #     nn.Conv2d(2, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.MaxPool2d(2),
        #     # (4, 4)
            
        #     nn.Flatten(),
        #     nn.Linear(4*4*256, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            
        #     nn.Linear(512, 10)
        # )
        
        
    def forward(self, x):
        return self.cnn(x)
        