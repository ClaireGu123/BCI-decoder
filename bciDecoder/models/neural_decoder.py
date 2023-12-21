import numpy as np
import pandas as pd

from torch import nn
import numpy as np
import pandas as pd

from torch import nn
from models.DeBCI import DeBCIForCTC


class NeuralDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_classes= config.model.n_classes
                

        self.flat = nn.Flatten(0)
        
        self.model = DeBCIForCTC(config)


    def forward(self, hidden_states,):
        output = self.model(hidden_states)
        return output