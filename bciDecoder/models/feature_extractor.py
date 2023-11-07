import numpy as np
import pandas as pd

from torch import nn

class FeatureExtractor(nn.Module):

    """
    (place holder) Feature extractor for the neural decoder model"""
    def __init__(self, config):
        super().__init__()
        channels= config.model.channels
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Linear(in_features=channels, out_features=channels),
            nn.GELU(),
            nn.Linear(in_features=channels, out_features=channels),
            nn.GeLU(),
        )


    def forward(self, x):
        return self.extractor(x)