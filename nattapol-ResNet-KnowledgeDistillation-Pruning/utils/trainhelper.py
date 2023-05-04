import torch
import torch.nn as nn
from models.resnet10 import resnet10
from models.simvit import simvit
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision.models import efficientnet_b0
from torchvision.models import efficientnet_b1
from dataset import *
from torchvision.ops import Conv2dNormActivation
from torch.utils.data import DataLoader
import math
from utils.loss.distillation import DistillationLoss

def find_abs_angle_difference(a, b):
    cos_theta = torch.cos(a/180 * math.pi) * torch.cos(b/180 * math.pi) 
    theta = torch.acos(cos_theta)
    return float(torch.sum(torch.abs(theta * 180 / math.pi)))

def find_jitter_metric(a:torch.Tensor,b:torch.Tensor) -> float:
    MSS = (torch.square(a) + torch.square(b)) /2
    return float(torch.sum(MSS))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(data_train, trainloader, 
                data_test, testloader, epoch, 
                model, optimizer, train_config):
    criterion = train_config['loss']
    running_loss = 0
    for i, data in enumerate(trainloader):
        images, labels = data
        yaws, pitchs = labels[0], labels[1]
        gts = torch.Tensor([[yaws[i], pitchs[i]] for i in range(images.size(0))])
        images = images.to(train_config['device'])
        gts = torch.Tensor(gts).to(train_config['device'])
        optimizer.zero_grad()
        if train_config['model'] in ['resnet10', 'resnet10+']:
            outputs, ifeature_1, ifeature_2, ifeature_3, ifeature_4 = model(images)
        else:
            outputs = model(images)
        if str(train_config['loss']) == "DistillationLoss()":
            model_teacher = resnet10()
            model_teacher.load_state_dict(torch.load(train_config['teacher_weight_path'], map_location=train_config['device']))
            model_teacher.train(False)
            model_teacher.to(train_config['device'])
            criterion.device = train_config['device']
            _, gtfeature1, gtfeature2, gtfeature3, gtfeature4 = model_teacher(images)
            loss = criterion(outputs, gts, [ifeature_1, ifeature_2], [gtfeature1, gtfeature2])
        else:
            loss = criterion(outputs, gts)
        del images
        del gts
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return float(running_loss / trainloader.__len__())

def test_epoch(data_test, testloader, model, epoch, train_config):
    running_error = 0
    for i, data in enumerate(testloader):
        images, labels = data
        yaws, pitchs = labels
        gts = torch.Tensor([[yaws[i], pitchs[i]] for i in range(images.size(0))])
        images = images.to(train_config['device'])
        gts = torch.Tensor(gts).to(train_config['device'])
        if train_config['model'] in ['resnet10', 'resnet10+']:
            outputs, _, _, _, _ = model(images)
        else:
            outputs = model(images)
        dif_y = outputs[:, 0] - gts[:, 0]
        dif_x = outputs[:, 1] - gts[:, 1]
        running_error += find_abs_angle_difference(dif_y, dif_x)
        del images
    return float(running_error / data_test.__len__())

def GetModel(train_config):
    model_name = train_config['model'] if train_config['model'][-1] != '+' else train_config['model'][:-1]
    out_channels = 2
    
    if train_config['model'][-1] == '+' ^ train_config['loss'] == DistillationLoss():
        raise Exception("Distillation model if only if using DistillationLoss()")
    
    if str(train_config['loss']) in ['GaussianLoss()', 'LogitLoss()']:
        out_channels = 1000
        
    if (model_name == 'resnet10'):
        model = resnet10(channels=64, out_channels=out_channels, input_size=train_config['res'])
        
    elif (model_name == 'resnet18'):
        assert train_config['res'] == (224,224), f'input size not supported'
        model = resnet18()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        model.fc = nn.Linear(512, out_channels)
        
    elif (model_name == 'resnet50'):
        assert train_config['res'] == (224,224), f'input size not supported'
        model = resnet50()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        model.fc = nn.Linear(512, out_channels)
        
    elif (model_name == 'EfficientNet_b0'):
        assert train_config['res'] == (224,224), f'input size not supported'
        model = efficientnet_b0()
        model.features[0] = Conv2dNormActivation(
                                1, 32, kernel_size=3, stride=2, 
                                norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU
                            )
        model.classifier = nn.Sequential(
                                nn.Dropout(p=0.1, inplace=True),
                                nn.Linear(1280, out_channels, bias=True)
                            )
        
    elif (model_name == 'EfficientNet_b1'):
        assert train_config['res'] == (224,224), f'input size not supported'
        model = efficientnet_b1()
        model.features[0] = Conv2dNormActivation(
                                1, 32, kernel_size=3, stride=2, 
                                norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU
                            )
        model.classifier = nn.Sequential(
                                nn.Dropout(p=0.1, inplace=True),
                                nn.Linear(1280, out_channels, bias=True)
                            )
    
    elif (model_name == 'SimVit'):
        assert train_config['res'] == (224,224), f'input size not supported'
        model = simvit()
    
    else: 
        raise Exception(f"{model_name} not founded")
        
    if train_config['model'][-1] == '+':
        return model, torch.load(train_config['teacher_weight_path'])
    else:
        return model, None
    
def GetDataset(train_config):
    if train_config['dataset'] == 'Columbia':
        idxList = [i for i in range(56)]
        testIdx = tuple(train_config['testIdx'])
        trainIdx = set(idxList) - set(tuple(testIdx))
        
        data_train = ColumbiaFaceDataset('/root/Columbia Gaze Data Set', 
                                         trainIdx, gray=True, 
                                         augmentation=train_config['augmentation'])
        data_test = ColumbiaFaceDataset('/root/Columbia Gaze Data Set', testIdx, 
                                        gray=True, augmentation=False)
        
        trainloader = DataLoader(data_train, batch_size=train_config['batch_size'], 
                                 shuffle=True, num_workers=2)
        testloader = DataLoader(data_test, batch_size=train_config['batch_size'], 
                                shuffle=False, num_workers=2)
        return data_train, data_test, trainloader, testloader
        
    elif train_config['dataset'] == 'Mpiigaze':
        idxList = [i for i in range(14)]
        testIdx = tuple(train_config['testIdx'])
        trainIdx = set(idxList) - set(tuple(testIdx))
        
        data_train = MpiigazeFaceDataset('/root/MPIIGaze_dataset/face/MPIIFaceGaze.h5', 
                                         trainIdx, 
                                         augmentation=train_config['augmentation'])
        data_test = MpiigazeFaceDataset('/root/MPIIGaze_dataset/face/MPIIFaceGaze.h5', 
                                        testIdx, augmentation=False)
        
        trainloader = DataLoader(data_train, batch_size=train_config['batch_size'], 
                                 shuffle=True, num_workers=2)
        testloader = DataLoader(data_test, batch_size=train_config['batch_size'], 
                                shuffle=False, num_workers=2)
        return data_train, data_test, trainloader, testloader
    
    else: raise Exception(f"{train_config['dataset']} not founded")