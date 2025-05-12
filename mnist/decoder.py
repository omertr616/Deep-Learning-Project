import torch
import torch.nn as nn




class decoder_mnist(nn.Module):
    def __init__(self):
        super().__init__()

        # self.rev_cnn = torch.nn.Sequential(
        #     nn.ConvTranspose2d(32, 256, kernel_size=5, padding=0, stride=2),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),

        #     nn.ConvTranspose2d(256, 1, kernel_size=7, padding=3),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(1),
        #     # nn.Upsample(scale_factor=4, mode='bilinear'),
        #     nn.Sigmoid()
        # )
        
        # self.rev_cnn = nn.Sequential(
        #     nn.ConvTranspose2d(32, 256, kernel_size=5, padding=0, stride=2),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),

        #     nn.ConvTranspose2d(256, 256, kernel_size=7, padding=3),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),  # (32, 12, 12)

        #     nn.ConvTranspose2d(256, 1, kernel_size=7, padding=3),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(1),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),  # (1, 24, 24)

        #     nn.Sigmoid()
        # )
        
        # self.rev_cnn = nn.Sequential(
        #     nn.Linear(128, 196),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(196),
        #     nn.Unflatten(1, (1, 14, 14)),
        #     nn.ConvTranspose2d(1, 256, kernel_size=5, padding=2, stride=1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),
        #     nn.ConvTranspose2d(256, 256, kernel_size=7, padding=3, stride=2),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),
        #     nn.ConvTranspose2d(256, 1, kernel_size=8, padding=3, stride=1),
        #     nn.ReLU(True),
        #     # nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )
        self.rev_cnn = nn.Sequential(
            nn.Linear(128, 196),
            nn.BatchNorm1d(196),
            nn.ReLU(),
            nn.Unflatten(1, (1, 14, 14)),
            
            nn.ConvTranspose2d(1, 256, kernel_size=7, padding=3, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 256, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 1, kernel_size=5, padding=2, stride=1),
            nn.Sigmoid()
            # nn.Tanh(),
        )
            
    def forward(self, x):
        return self.rev_cnn(x)
