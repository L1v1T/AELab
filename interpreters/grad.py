import torch
import torch.nn as nn
from copy import deepcopy

class GRAD(nn.Module):
    def __init__(self, model):
        super(GRAD, self).__init__()
        self.model = deepcopy(model)

    def forward(self, x, y=None):
        if not x.requires_grad:
            x.requires_grad_(True)
        
        output = self.model(x)

        if y == None:
            y = output.argmax(dim=1, keepdim=True)
        
        # scores = output.gather(1, y.view(-1, 1)).squeeze()
        scores = output.gather(1, y.view(-1, 1))
        
        scores.backward(torch.ones_like(scores))
        saliency_map = x.grad.data
        saliency_map = saliency_map.unsqueeze(dim=1)

        saliency_map = saliency_map.abs()
        saliency_map, _ = torch.max(saliency_map, dim = 1)

        return saliency_map