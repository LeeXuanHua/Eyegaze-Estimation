from data_loader import data_loader_from_csv
import numpy as np
from model import UnsupervisedGazeNetwork
from loss import Loss
import torch
import torch.optim as optim

if __name__ == "__main__":
    model = UnsupervisedGazeNetwork()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = Loss()
    batch_size = 32

    for epoch in range(2000):

        # only pair up images from the same person (same number at the end of file name)
        # file is in bst Google drive
        images, gts = data_loader_from_csv(('{path_to_gts}', '{path_to_images}'))
        for i in range(images.size(0)//batch_size):
            index_1, index_2 = np.random.randint(0, images.size(0)-batch_size), np.random.randint(0, images.size(0)-batch_size)
            image_1, image_2 = images[index_1:index_1+batch_size].view(-1, 3, 36, 60), images[index_2:index_2+batch_size].view(-1, 3, 36, 60)

            output, feature_i, feature_o = model(image_1, image_2)
            loss = criterion(output, image_2, feature_i, feature_o)
            loss.backward()
            optimizer.step()
            print(f'{epoch+1}, \nloss: {loss}, loss_feature: {criterion.coef_perceptual*criterion.loss__feature}, loss_pixel: {criterion.loss__pixel}')