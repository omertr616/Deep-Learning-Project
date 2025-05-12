import torch
import torch.nn as nn

class encoder_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### accuracy 54 ###
        
        # dropout = 0
        # self.cnn = nn.Sequential(
        #     # (32, 32)
        #     nn.Conv2d(3, 256, kernel_size=7, padding=3, stride=1),
        #     # (32, 32)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.Conv2d(256, 256, kernel_size=7, padding=3, stride=1),
        #     nn.MaxPool2d(2),
        #     # (16, 16)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.Conv2d(256, 2, kernel_size=5, padding=2, stride=1),
        #     nn.MaxPool2d(2),
        #     # (8, 8)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(2),
        #     nn.Dropout2d(dropout),
        #     nn.Flatten()
        # )
        
        
        ### accuracy 53 ###
        
        # dropout = 0
        # self.cnn = nn.Sequential(
        #     # (32, 32)
        #     nn.Conv2d(3, 256, kernel_size=7, padding=3, stride=1),
        #     # (32, 32)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.Conv2d(256, 256, kernel_size=7, padding=3, stride=1),
        #     nn.MaxPool2d(2),
        #     # (16, 16)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.Conv2d(256, 1, kernel_size=5, padding=2, stride=1),
        #     # (16, 16)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(1),
        #     nn.Dropout2d(dropout),
        #     nn.Flatten(),
        #     nn.Linear(16*16, 128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128)
        # )
        
        
        # dropout = 0
        # self.cnn = nn.Sequential(
        #     # (32, 32)
        #     nn.Conv2d(3, 256, kernel_size=7, padding=3, stride=1),
        #     # (32, 32)
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.Conv2d(256, 512, kernel_size=7, padding=3, stride=1),
        #     nn.MaxPool2d(2),
        #     # (16, 16)
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.Conv2d(512, 2, kernel_size=5, padding=2, stride=1),
        #     nn.MaxPool2d(2),
        #     # (8, 8)
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        
        ## 62 ###
        # dropout = 0
        self.cnn = nn.Sequential(
            # (32, 32)
            nn.Conv2d(3, 256, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            # (16, 16)
            nn.Conv2d(256, 256, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            # (8, 8)
            nn.Conv2d(256, 2, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # dropout = 0
        # self.cnn = nn.Sequential(
        #     # (32, 32)
        #     nn.Conv2d(3, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.MaxPool2d(2),
        #     # (16, 16)
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.MaxPool2d(2),
        #     # (8, 8)
        #     nn.Conv2d(256, 2, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
            
    def forward(self, x):
        return self.cnn(x)
    