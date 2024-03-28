from torch import nn

from models.DeBCI import DeBCIForCTC, DeBCIModel
from transformers import BertModel, BertTokenizer

class NeuralDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = DeBCIForCTC(config)

    def forward(self, hidden_states,):
        output = self.model(hidden_states)
        return output

