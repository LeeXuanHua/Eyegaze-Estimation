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
import argparse

PROJECT_PATH = os.getenv('PROJECT_PATH')

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help="resnet10 | resnet18", default='resnet10')
parser.add_argument('--lr', type=float, help="Learning rate", default=1e-5)
parser.add_argument('--dataset', type=str, help="Mpiigaze | Columbia", default='Columbia')
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--augmentation', action=argparse.BooleanOptionalAction)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--id_test', type=int, default=0)
parser.add_argument('--prune_ratio', type=float, default=0.2)

args = parser.parse_args()

# 5-fold split for Columbia
fold = []
fold.append(tuple((7, 11, 53, 18, 17, 29, 30, 33, 22, 46, 32)))
fold.append(tuple((51, 1, 45, 20, 25, 2, 14, 37, 10, 27, 6)))
fold.append(tuple((55, 9, 8, 49, 16, 4, 28, 52, 19, 39, 23)))
fold.append(tuple((50, 47, 5, 54, 34, 40, 36, 13, 38, 42, 41)))
fold.append(tuple((21, 0, 12, 15, 48, 24, 44, 35, 43, 26, 31, 3)))

if args.dataset == 'Mpiigaze':
    if args.id_test > 14 or args.id_test < 0:
        raise Exception(f"Test id should be between [0,14] for Mpiigaze")
elif args.dataset=='Columbia':
    if args.id_test > 4 or args.id_test < 0:
        raise Exception(f"Test id should be between [0,4] for Columbia")

train_config = {
    'model': args.model,
    'epoch': args.epoch,
    'res': (224, 224),
    'learning_rate': args.lr,
    'dataset': args.dataset,
    'batch_size': args.batch_size,
    'augmentation': args.augmentation,
    'loss': nn.MSELoss(),
    'device': torch.device(args.device),
}

def main(train_config):
    minerror = 10
    model, model_teacher = GetModel(train_config)
    if train_config['weight_prune']:
        model.load_state_dict(torch.load(train_config['teacher_weight_path']))
        model = prune(model, prune_ratio=args.prune_ratio)
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
        epoch_summary =  f'[{args.model}]\t{epoch+1}\t\t{str(avg_loss)[:7]}'
        epoch_summary += f'\t\t{str(float(train_error))[:7]}'
        epoch_summary += f'\t\t{str(float(test_error))[:7]}'
        epoch_summary += f'\t\t{int(time.time()-starttime) // 60}'
        print(epoch_summary)
    
    if train_config['weight_prune']:
        file_path = f"{PROJECT_PATH}/results/pretrained/{args.model}+P"
    else:
        file_path = f"{PROJECT_PATH}/results/pretrained/{args.model}"
        
    file_name = file_path + f'/{args.model}-{args.id_test}-{args.dataset}'
    print(f'[{args.model}] saved in {file_path}.pt')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    torch.save(model.state_dict(), file_name+'.pt')
    teacher_path  = file_name
    model_summary = [train_config['model'], train_config['learning_rate'], 
                        train_config['batch_size'], train_config['epoch'],  
                        file_name+'.pt', str(float(test_error))[:5], 
                        train_config['dataset']]
    model_name = train_config['model']
    with open(f'/root/_KD/Result_Face/{model_name}P.csv', 'a') as f:
        writerObj = writer(f)
        writerObj.writerow(model_summary)
    return teacher_path
    
if __name__ == '__main__':
    if train_config['dataset'] == 'Mpiigaze':
        train_config['testIdx'] = tuple((args.id_test,))
    else:
        train_config['testIdx'] = fold[args.id_test]
    
    print("(1) Train without Knowledge Distillation")
    train_config['weight_prune'] = False
    print(train_config)
    teacher_weight_path = main(train_config)
    
    print("(2) Train with Knowledge Distillation")
    train_config['model'] += '+'
    train_config['loss'] = DistillationLoss()
    train_config['teacher_weight_path'] = teacher_weight_path
    train_config['weight_prune'] = True
    train_config['epoch'] = int(train_config['epoch']/3)
    _ = main(train_config)
    print(" \t-- Training Finished -- ")