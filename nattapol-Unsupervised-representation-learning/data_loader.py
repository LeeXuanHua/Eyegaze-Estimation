import numpy as np
import torch

i=0 
# i in range [0,55]

path_gts = "/content/drive/MyDrive/data_gt_{i}.csv"
path_images = "/content/drive/MyDrive/data_image_{i}.csv"

def data_loader_from_csv(path = (path_gts, path_images)):
  my_data = np.genfromtxt(path[0], delimiter=',')
  gts = np.array(my_data).reshape((-1, 2))
  gts = torch.from_numpy(gts).to(torch.float)
  print(f"got ground truth with size: {gts.shape}")
  my_data = np.genfromtxt(path[1], delimiter=',')
  images = np.array(my_data).reshape((-1, 3, 36, 60))
  images = torch.from_numpy(images).to(torch.float)
  images = images / torch.max(torch.abs(images))
  print(f"got images with size: {images.shape}")
  return images, gts