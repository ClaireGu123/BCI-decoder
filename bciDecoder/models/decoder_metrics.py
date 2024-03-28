from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

import torch

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class CLIPLoss(Metric):

    def __init__(self, ignored_class, output_transform=lambda x: x, device="cpu"):
        self.ignored_class = ignored_class
        super(CLIPLoss, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        super(CLIPLoss, self).reset()

    @reinit__is_reduced
    def update(self, output):

        self._loss = output.item()

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        return self._loss