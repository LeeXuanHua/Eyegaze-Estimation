from model import GazeRepresentationLearning
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils import find_abs_angle_difference
from data_loader import data_loader_from_csv
import torch.optim as optim

number_of_test = 5
number_of_epoch = 200
batch_size = 96
show_images = False

# load all data from all files [0,55]
# file is in bst Google drive
images, gts = data_loader_from_csv({path_to_gt},{path_to_images})

for __ in range(number_of_test):
    assert type(images).__name__ == "Tensor", "wrong images format"
    assert type(gts).__name__ == "Tensor", "wrong gts format"
    model = GazeRepresentationLearning()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    for epoch in range(number_of_epoch):
        for i in range(40):
            image_1 = images[batch_size*i:batch_size*(i+1)].view(batch_size, 3, 36, 60)
            outputs = model(image_1)
            gt = gts[batch_size*i:batch_size*(i+1)]
            if show_images and np.random.randint(0, 10000) > 9990:
                plt.imshow(image_1[0,0,:,:])
                plt.title(f'ground truth = {gt}, output = {outputs}')
                plt.show()
            if epoch % 20 == 10 and i == 10:
                for g in optimizer.param_groups:
                    g['lr'] /= 2
                print("reduce lr")
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
        if epoch%10 == 0:
            print(f'test: {__+1}, epoch: {epoch+1}, loss: {loss}')
    outputs = model(images[3840:])
    dif = gts[3840:] - outputs
    yaw = dif[:, 0]
    pitch = dif[:, 1]
    val = find_abs_angle_difference(yaw, pitch)
    print(f"{__ + 1}: got mean angle error: {torch.sum(val/outputs.size(0))}")