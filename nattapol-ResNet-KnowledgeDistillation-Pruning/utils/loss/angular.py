import torch
import torch.nn as nn
import math

class AngularLoss(nn.Module):
    """
        pred: Tensor [N]
        gt: tensor [N]
    """
    def __init__(self, coef_mse = 1, coef_abs_ang= 0.05) -> None:
        super(AngularLoss, self).__init__()
        self.coef_mse = coef_mse
        self.coef_abs_ang = coef_abs_ang
    def forward(self, pred, gt):
        v = torch.abs(pred[:, 0] - gt[:, 0])
        h = torch.abs(pred[:, 1] - gt[:, 1])
        absAng = torch.sum(2 - torch.cos(v * 180/math.pi) - torch.cos(h * 180/math.pi))
        mse = nn.MSELoss()(pred, gt)
        return absAng * self.coef_abs_ang + mse * self.coef_mse

class SineLoss(nn.Module):
    """
        pred: Tensor [N]
        gt: tensor [N]
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, pred, gt):
        pred_sin = torch.sin(pred * 180/math.pi) * 0.5 + 0.5
        pred_cos = torch.cos(pred * 180/math.pi) * 0.5 + 0.5
        gt_sin = torch.sin(gt * 180/math.pi) * 0.5 + 0.5
        gt_cos = torch.cos(gt * 180/math.pi) * 0.5 + 0.5
        return 5 * (nn.MSELoss()(pred_sin, gt_sin) + nn.MSELoss()(pred_cos, gt_cos)) + nn.MSELoss()(pred, gt)
    
class SectorLoss(nn.Module):
    """
        pred: Tensor [N]
        gt: tensor [N]
    """
    def __init__(self) -> None:
        super().__init__()
    def forward(self, pred, gt):
        dv = (pred[:, 0] - gt[:, 0]) * 180/math.pi
        dh = (pred[:,1] - gt[:, 1]) * 180/math.pi
        return torch.sqrt(torch.sum(torch.abs(torch.sin(dv/2)) + torch.abs(torch.sin(dh/2))))
        