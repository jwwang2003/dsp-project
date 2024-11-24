import torch.nn as nn
from reformer_pytorch import Reformer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias): 
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, activation=True):
        super(ConvUpsampleBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.activation = activation
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        if self.activation:
            out = self.relu(out)
        return out

class Hybrid(nn.Module):
    def __init__(self, input_samples: int, n_classes: int, debug = False):
        super().__init__()
        
        self.debug = debug
        
        self.encoder = nn.Sequential(
        ConvBlock(2, 128, n_classes, 6, False),
        nn.MaxPool1d(2),
        ConvBlock(128, 256, n_classes, 6, False),
        nn.MaxPool1d(2),    
        ConvBlock(256, 256, n_classes, 6, False),
        ConvBlock(256, 256, n_classes, 6, False),)
        
        self.reformer = Reformer(dim = 256, depth = 2,  heads = 8, lsh_dropout = 0.1, causal = True, bucket_size = 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*int(input_samples/4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, n_classes),)

        self.decoder = nn.Sequential(
        ConvUpsampleBlock(256, 128, n_classes, 6, False),
        nn.Upsample(scale_factor=2),
        ConvUpsampleBlock(128, 128, n_classes, 6, True),
        nn.Upsample(scale_factor=2),
        ConvUpsampleBlock(128, 64, n_classes, 6, True),
        ConvUpsampleBlock(64, 2, n_classes, 6, True,activation=False),)
        
    def forward(self, input_):
        z = self.encoder(input_)
        
        if self.debug: print(z.shape)
        
        recon = self.decoder(z)
        
        if self.debug: print(recon.shape)
        
        z = self.reformer(z.permute(0,2,1))
        y = self.classifier(z.permute(0,2,1))
        return y,recon