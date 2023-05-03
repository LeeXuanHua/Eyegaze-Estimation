from utils.dataset import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from utils.models.resnet import resnet10, resnet18
from csv import writer
from utils.trainhelper import *
from utils.testhelper import *
from utils.weight_prune import *
from utils.loss.distillation import DistillationLoss
from utils.loss.feature import PearsonSimilarityLoss

# index for 5-fold cross validation for Colubmbia Gaze
fold1 = (7, 11, 53, 18, 17, 29, 30, 33, 22, 46, 32)
fold2 = (51, 1, 45, 20, 25, 2, 14, 37, 10, 27, 6) 
fold3 = (55, 9, 8, 49, 16, 4, 28, 52, 19, 39, 23) 
fold4 = (50, 47, 5, 54, 34, 40, 36, 13, 38, 42, 41) 
fold5 = (21, 0, 12, 15, 48, 24, 44, 35, 43, 26, 31, 3) 

train_config = {
    'model': 'ResNet18', 
    'epoch': 40, 
    'res': (224, 224), 
    'learning_rate': 1e-4,
    'dataset': 'Columbia',
    'batch_size': 32,
    'augmentation': True,
    'testIdx': fold1,
    'loss': nn.MSELoss(),
    # 'loss': DistillationLoss(),
    'device': torch.device("cuda:0"),
}

def main(train_config):
    minerror = 10
    model, model_teacher = GetModel(train_config)
    if train_config['weight_prune']:
        model.load_state_dict(torch.load(train_config['teacher_weight_path']))
    model = prune(model, prune_ratio=0.1)
    train_config['model_teacher'] = model_teacher
    data_train, data_test, trainloader, testloader = GetDataset(train_config)
    print(f'number of training samples = {len(data_train)}')
    optimizer = optim.Adam([
        {'params' : model.parameters()},
        ], lr = train_config['learning_rate'], weight_decay=0.01)
    starttime = time.time()
    number_of_epoch = train_config['epoch']
    model.to(train_config['device'])
    atepoch = 0
    for epoch in range(number_of_epoch):
        model.train(True)
        avg_loss = train_epoch(data_train, trainloader, 
                               data_test, testloader, 
                               epoch, model, optimizer, train_config)
        model.train(False)
        test_error = test_epoch(data_test, testloader, model, epoch, train_config)
        
        minerror = min(minerror, test_error)
        atepoch = epoch if (minerror == test_error) else atepoch
        train_error = test_epoch(data_train, trainloader, model, epoch, train_config)
        print(f'epoch: {epoch+1} loss= {str(avg_loss)[:5]} train_error = {str(float(train_error))[:5]} \
            test_error= {str(float(test_error))[:5]} time = {int(time.time()-starttime) // 60}')
        if train_config['pearson_regularize']:
            print(f'{str(float(PearsonSimilarityLoss()(model)))}')
    
    file_name = f"/root/_KD/Result_Face/{train_config['model']}{str(float(test_error))}.pt"
    torch.save(model.state_dict(), file_name)
    teacher_path  = file_name
    model_summary = [train_config['model']+'P', train_config['learning_rate'], 
                        train_config['batch_size'], train_config['epoch'], 
                        False, count_parameters(model), 
                        file_name, str(float(test_error))[:5], 
                        train_config['dataset'], len(data_train), 
                        train_config['dataset'], str(int(time.time()-starttime)/60)[:5],
                        str(train_config['loss']), str(train_config['testIdx'][0]), minerror, atepoch]
    model_name = train_config['model']
    with open(f'/root/_KD/Result_Face/{model_name}P.csv', 'a') as f:
        writerObj = writer(f)
        writerObj.writerow(model_summary)
    return teacher_path
    
if __name__ == '__main__':
    
    model_name = train_config['model']
    df = pd.read_csv(f'/root/_KD/Result_Face/{model_name}.csv')
    df = df[df['foldid']==str(train_config['testIdx'][0])]
    df = df[df['train dataset']==train_config['dataset']]
    train_config['teacher_weight_path'] = df['path to pretrain'][list(df.index.values)[0]]
    train_config['loss'] = DistillationLoss()
    train_config['model'] = 'ResNet18+'
    print(train_config)
    _ = main(train_config)