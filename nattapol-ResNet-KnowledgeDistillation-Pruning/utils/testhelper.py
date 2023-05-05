import torch
import torch.nn as nn
from dataset import *
import math

def find_abs_angle_difference(a, b):
    cos_theta = torch.cos(a/180 * math.pi) * torch.cos(b/180 * math.pi) 
    theta = torch.acos(cos_theta)
    return float(torch.sum(torch.abs(theta * 180 / math.pi)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_epoch(data_test, testloader, model, epoch, train_config):
    running_error = 0
    for i, data in enumerate(testloader):
        images, labels = data
        yaws, pitchs = labels
        gts = torch.Tensor([[yaws[i], pitchs[i]] for i in range(images.size(0))])
        images = images.to(train_config['device'])
        gts = torch.Tensor(gts).to(train_config['device'])
        if train_config['model'] in ['ResNet10', 'ResNet10+']:
            outputs, _, _, _, _ = model(images)
        else:
            outputs = model(images)
        dif_y = outputs[:, 0] - gts[:, 0]
        dif_x = outputs[:, 1] - gts[:, 1]
        running_error += find_abs_angle_difference(dif_y, dif_x)
        del images
    return float(running_error / data_test.__len__())
