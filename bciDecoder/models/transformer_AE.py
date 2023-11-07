import numpy as np
import pandas as pd

from torch import nn
from models.feature_extractor import FeatureExtractor

class NeuralDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        frame_lens = config.model.num_frames
        n_classes= config.model.n_classes
                

        self.flat = nn.Flatten(0)
        
        self.feature_extractor = FeatureExtractor(config)

        self.linear_proj = nn.Linear(frame_lens, n_classes)

    def encoder(self, x):
        #x = self.flat(x)
        # x = self.bn(x.permute(0,2,1))
        z, _ = self.lstm_encode(x.permute(0,2,1))
        # z = self.feature_extractor(x)
        return z
    
    def decoder(self, x):
        x, _ = self.lstm_decode(x)
        output = self.linear_proj(x)
        return output

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y