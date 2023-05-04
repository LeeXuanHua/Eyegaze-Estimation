import torch
import torch.nn as nn
class PearsonSimilarityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, model):
        running_loss = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) and module.weight.size(1)>10 and module.weight.size(3)>=3:
                weight_flatten = module.weight.view(module.weight.size(0), -1)
                matrix = torch.corrcoef(weight_flatten)
                correlation = torch.norm(matrix) - torch.norm(torch.diag(torch.ones(matrix.size(), dtype=module.weight.dtype)))
                running_loss += correlation
        return running_loss