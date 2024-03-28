import torch
from torch import nn

import ignite.distributed as idist

from models.DeBCI import DeBCIModel


device = idist.device()

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ecog_encoder =  DeBCIModel(config)


    def forward(self, hidden_states, text_embeds):
        ecog_embeds = torch.mean(self.ecog_encoder(hidden_states)[0], dim=1) # [batch_size, T,768] # aggregate to 768
        cross_cor = torch.matmul(torch.permute(text_embeds,(1,0)), ecog_embeds) / text_embeds.shape[0]
        c_diff = (cross_cor - torch.eye(text_embeds.shape[1]).to(device)).pow(2)
        scaled = c_diff.mul_(self.config.model.reg)
        loss = torch.diagonal(c_diff).sum() + scaled.sum() - torch.diagonal(scaled).sum()
        
        return loss



