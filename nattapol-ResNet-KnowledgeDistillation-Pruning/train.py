from utils.dataset import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from random import sample
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import time
from utils.models.resnet import resnet10, resnet18
from csv import writer
from utils import *
from utils.loss.angular import AngularLoss
from utils.loss.distillation import DistillationLoss
from utils.trainhelper import *
from utils.testhelper import *

"""
ENUM train_config {
    'model': resnet10 | resnet18 | resnet50 | EfficientNet_b0 | EfficientNet_b1 | SimVit,
    'epoch': int,
    'res': Tuple[int, int],
    'learning_rate': float,
    'dataset': 'Mpiigaze' | 'Columbia',
    'batch_size': int,
    'augmentation': bool,
    'teacher_weight_path'?: '/path/to/teacher/'  # in case of loss == DistillationLoss()
    'testIdx': tuple[..., int],
    'loss': nn.MSELoss() | DistillationLoss() | AngularLoss(),
    'device': torch.device("your cuda/cpu device"),
}
"""

train_config = {
    'model': 'resnet10',
    'epoch': 40,
    'res': (224, 224),
    'learning_rate': 1e-5,
    'dataset': 'Mpiigaze',
    'batch_size': 32,
    'augmentation': True,
    'teacher_weight_path': '/path/to/teacher/model',
    'testIdx': tuple((14,)),
    'loss': nn.MSELoss(),
    'device': torch.device("cuda:0"),
}

def main(train_config):
    model, model_teacher = GetModel(train_config)
    train_config['model_teacher'] = model_teacher
    data_train, data_test, trainloader, testloader = GetDataset(train_config)
    print(f'number of train samples = {len(data_train)}')
    
    optimizer = optim.Adam([
        {'params' : model.parameters()},
        ], lr = train_config['learning_rate'])
    model.train()
    starttime = time.time()
    
    number_of_epoch = train_config['epoch']
    model.to(train_config['device'])
    for epoch in range(number_of_epoch):
        model.train(True)
        avg_loss = train_epoch(data_train, trainloader, 
                               data_test, testloader, 
                               epoch, model, optimizer, train_config)
        model.train(False)
        test_error = test_epoch(data_test, testloader, model, epoch, train_config)
        train_error = test_epoch(data_train, trainloader, model, epoch, train_config)
        
        epoch_summary =  f'epoch: {epoch+1} loss: {str(avg_loss)[:5]}'
        epoch_summary += f' train error: {str(float(train_error))[:5]}'
        epoch_summary += f' test_error= {str(float(test_error))[:5]}'
        epoch_summary += f' time: {int(time.time()-starttime) // 60}'
        print(epoch_summary)

    file_name = f"/path/to/model"
    torch.save(model.state_dict(), file_name)
    model_summary = [train_config['model'], train_config['learning_rate'], 
                        train_config['batch_size'], train_config['epoch'], 
                        False, count_parameters(model), 
                        file_name, str(float(test_error))[:5], 
                        train_config['dataset'], len(data_train), 
                        train_config['dataset'], str(int(time.time()-starttime)/60)[:5],
                        str(train_config['loss']), str(train_config['testIdx'][0])]
    with open('/path/to/log/', 'a') as f:
        writerObj = writer(f)
        writerObj.writerow(model_summary)
    
if __name__ == '__main__':
    print(train_config)
    main(train_config)