import torch
import torch.nn as nn

class encoder_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.cnn = torch.nn.Sequential(
        #     nn.BatchNorm2d(1),
        #     nn.Conv2d(1, 256, kernel_size=7, padding=3),
        #     nn.MaxPool2d(4),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),
        #     # nn.Conv2d(256, 256, kernel_size=7, padding=3),
        #     # nn.MaxPool2d(2),
        #     # nn.ReLU(True),
        #     # nn.BatchNorm2d(256),
        #     nn.Conv2d(256, 32, kernel_size=5, padding=2),
        #     nn.MaxPool2d(3),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(32))
        
        # self.cnn = nn.Sequential(
        #     nn.BatchNorm2d(1),
        #     nn.Conv2d(1, 256, kernel_size=7, padding=0,stride=1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),
        #     nn.Conv2d(256, 256, kernel_size=5, padding=0,stride=1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(256),
        #     nn.Conv2d(256, 1, kernel_size=5, padding=0,stride=1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(1),
        #     nn.Flatten(),
        #     nn.Linear(196, 128),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(128)   
        # )
        # dropout = 0
        # self.cnn = nn.Sequential(
        #     # (28, 28)
        #     nn.Conv2d(1, 256, kernel_size=7, padding=3, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
            
        #     nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     # (14, 14)
        #     nn.Dropout2d(dropout),
            
        #     nn.Conv2d(256, 1, kernel_size=5, padding=2, stride=1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout),
        #     nn.Flatten(),
            
        #     nn.Linear(196, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )
        
        self.cnn = nn.Sequential(
            # (28, 28)
            nn.Conv2d(1, 256, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # (14, 14)
            
            nn.Conv2d(256, 1, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            
            nn.Linear(196, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # self.cnn = nn.Sequential(
        #     # (28, 28)
        #     nn.Conv2d(1, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
            
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     # (14, 14)
            
        #     nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(),
        #     nn.Flatten(),
            
        #     nn.Linear(196, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )
            
    def forward(self, x):
        return self.cnn(x)
        
    
# def train_batch_encoder_decoder_mnist(encoder, decoder, batch, optimizer, loss_fn):
#     encoder.train()
#     decoder.train()
#     optimizer.zero_grad()
#     z = encoder(batch)
#     rec_x = decoder(z)
#     loss = loss_fn(rec_x, batch)
#     loss.backward()
#     optimizer.step()
#     return loss

# def test_batch_encoder_decoder_mnist(encoder, decoder, batch, loss_fn):
#     encoder.eval()
#     decoder.eval()
#     with torch.no_grad():
#         z = encoder(batch)
#         rec_x = decoder(z)
#         loss = loss_fn(rec_x, batch)
#     return loss
