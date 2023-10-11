import torch
import torch.nn as nn


class Linear(nn.Linear):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
    ) -> None:
        super(nn.Linear, self).__init__()