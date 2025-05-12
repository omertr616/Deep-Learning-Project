import torch
import torch.nn as nn


class decoder_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        ### accuracy 54 ###
        # dropout = 0
        # self.rev_cnn = nn.Sequential(
        #     nn.Unflatten(1, (2, 8, 8)),
        #     nn.ConvTranspose2d(2, 256, kernel_size=5, padding=2, stride=2, output_padding=1),
        #     # (16, 16)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 256, kernel_size=7, padding=3, stride=2, output_padding=1),
        #     # (32, 32)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 3, kernel_size=7, padding=3, stride=1),
        #     nn.Sigmoid()
        # )
        
        # dropout = 0
        # self.rev_cnn = nn.Sequential(
        #     nn.Unflatten(1, (2, 8, 8)),
        #     nn.ConvTranspose2d(2, 256, kernel_size=5, padding=2, stride=2, output_padding=1),
        #     # (16, 16)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 3, kernel_size=7, padding=3, stride=2, output_padding=1),
        #     # (32, 32)
        #     nn.Sigmoid()
        # )
        
        ### accuracy 53 ###
        # dropout = 0
        # self.rev_cnn = nn.Sequential(
        #     nn.Linear(128, 16*16),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(16*16),
        #     nn.Unflatten(1, (1, 16, 16)),
        #     # (16, 16)
        #     nn.ConvTranspose2d(1, 256, kernel_size=5, padding=2, stride=1),
        #     # (16, 16)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 256, kernel_size=7, padding=3, stride=2, output_padding=1),
        #     # (32, 32)
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 3, kernel_size=7, padding=3, stride=1),
        #     nn.Sigmoid()
        # )
        
        # dropout = 0
        # self.rev_cnn = nn.Sequential(
        #     nn.Unflatten(1, (2, 8, 8)),
        #     nn.ConvTranspose2d(2, 512, kernel_size=5, padding=2, stride=1),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     # (16, 16)
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(512, 256, kernel_size=7, padding=3, stride=1),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     # (32, 32)
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 3, kernel_size=7, padding=3, stride=1),
        #     nn.BatchNorm2d(3),
        #     nn.Sigmoid()
        # )
        
        # dropout = 0
        # self.rev_cnn = nn.Sequential(
        #     nn.Unflatten(1, (2, 8, 8)),
        #     nn.ConvTranspose2d(2, 256, kernel_size=5, padding=2, stride=2, output_padding=1),
        #     # (16, 16)
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 256, kernel_size=7, padding=3, stride=2, output_padding=1),
        #     # (32, 32)
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.ConvTranspose2d(256, 3, kernel_size=7, padding=3, stride=1),
        #     nn.Sigmoid()
        # )
        
        
        ### accuracy 61 ###
        # dropout = 0
        # self.rev_cnn = nn.Sequential(
        #     nn.Unflatten(1, (2, 8, 8)),

        #     nn.ConvTranspose2d(2, 256, kernel_size=5, padding=2, stride=1),  
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),

        #     # (8,8)->(16,16) (reverse of MaxPool2d)
        #     nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),

        #     # (16,16)->(32,32) (reverse of MaxPool2d)
        #     nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout2d(dropout),

        #     nn.Conv2d(256, 3, kernel_size=7, padding=3, stride=1),
        #     nn.Tanh()
        #     # nn.Sigmoid()
        # )
        
        
        # dropout = 0
        # self.rev_cnn = nn.Sequential(
        #     nn.Unflatten(1, (2, 8, 8)),

        #     nn.ConvTranspose2d(2, 256, kernel_size=3, padding=1, stride=1),
        #     # (8, 8)
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),

        #     nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     # (16, 16)
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),

        #     nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     # (32, 32)
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),

        #     nn.Conv2d(256, 3, kernel_size=3, padding=1, stride=1),
        #     nn.Tanh()
        # )
        
        self.rev_cnn = nn.Sequential(
            nn.Unflatten(1, (2, 8, 8)),

            nn.ConvTranspose2d(2, 256, kernel_size=5, padding=2, stride=1),
            # (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=2, padding=3, output_padding=1),
            # (16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=2, padding=3, output_padding=1),
            # (32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 3, kernel_size=7, padding=3, stride=1),
            nn.Tanh()
        )
        
            
    def forward(self, x):
        return self.rev_cnn(x)
    