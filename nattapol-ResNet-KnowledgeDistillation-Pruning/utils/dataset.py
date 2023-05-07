import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import h5py
from random import shuffle
import math

from utils.augmentation import *

class ColumbiaFaceDataset(Dataset):
    # Columbia Gaze Dataset (INPUT: PREPROCESSED CROPPED FACE IMAGE)
    def __init__(self, path: str, 
                 person_id: tuple[int, ...], 
                 augmentation=1, outSize=(224,224), 
                 gray=True):
        self.path = path
        self.gray = gray
        self.augmentation = augmentation
        self.person_id = person_id
        self.C = 1 if gray else 3
        self.outSize = outSize
        self._augmentation()
        self.load_data()
        
    def _augmentation(self):
        if (self.augmentation):
            crop = RandomCrop(self.outSize, self.C)
            flip = RandomHorizontalFlip(self.C)
            blur = GaussianBlur(self.C)
            sharp = RandomAdjustSharpness(self.C)
            mixup = SelfMixup(self.outSize, self.C)
            mixup_2 = SelfMixup_2(self.outSize, self.C)
            self.trsfrm = [crop, flip, blur, sharp, mixup, mixup_2]
            
    def __len__(self) -> int:
        return (len(self.images))
    
    def load_data(self):
        self.files_path = []
        for each in os.listdir(self.path):
            if each[:2] == '00' and int(each[2:4]) in self.person_id:
                self.files_path.extend([f.path for f in os.scandir(self.path+'/'+each+'/') 
                                        if f.is_file() and f.path[-5:]=='H.jpg'])
                
        self.images, self.labels = [], []
        for person in self.files_path:
            with Image.open(person) as image:
                # since the face center is at the same place, we can 
                # extract the face by directly cropping at the same place.
                image = transforms.Resize(size=(864,576))(image)
                image = transforms.CenterCrop(size=(250,250))(image)
                image = transforms.Resize(size=(224,224))(image)
                
                if self.gray: 
                    image = image.convert("L")
                image = transforms.ToTensor()(image)
                image_processed = transforms.Normalize([0.5] * self.C, [0.5] * self.C)(image)
                
                yaw, pitch = self.read_angles(person)
                
                self.images.append(image_processed)
                self.labels.append([yaw, pitch])
                del image
                del image_processed
        
    def read_angles(self, name):
        filter = name.split("/")[-1].split("_")
        yaw = float(filter[3][:-1])
        pitch = float(filter[4].replace("H.jpg", ''))
        return yaw, pitch
    
    def __getitem__(
            self, 
            index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # refresh augmentation random parameter every image
        self._augmentation()
        if self.augmentation:
            idx = np.random.randint(len(self.trsfrm) + 1)
            if idx >= len(self.trsfrm):
                return self.images[index], self.labels[index]
            else :
                image, labels = self.images[index], self.labels[index]
                method = self.trsfrm[idx]
                image, vert, hor = method((image, labels[0], labels[1]))
                return image, [vert, hor]
        return self.images[index], self.labels[index]
    
class MpiigazeFaceDataset(Dataset):
    # MPIIFaceGaze Dataset (INPUT: PREPROCESSED CROPPED FACE IMAGE)
    def __init__(self, path: str, 
                 person_id: tuple[int, ...], gray=True,
                 augmentation=True, outSize=(224,224)):
        self.path = path
        self.images, self.labels = [], []
        self.C = 1 if gray else 3
        self.augmentation = augmentation
        self.person_id = person_id
        self.outSize = outSize
        self.gray = gray
        self._augmentation()
        self._load_data()

    def _load_data(self):
        for idx in self.person_id:
            person_id_str = str(idx)
            person_id_str = 'p0' + person_id_str if len(person_id_str) == 1 else 'p' + person_id_str
            for index in range(3000):
                with h5py.File(self.path, 'r') as f:
                    images = f.get(f'{person_id_str}/image/{index:04}')[()]
                    labels = f.get(f'{person_id_str}/gaze/{index:04}')[()] * 180/ math.pi
                labels[1] = -labels[1]
                
                preprocess = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Resize(size=self.outSize)])
                if self.gray:
                    preprocess = transforms.Compose([transforms.ToTensor(), 
                                                    transforms.Grayscale(),
                                                    transforms.Resize(size=self.outSize)])
                yaw, pitch = labels
                image_matrix = preprocess(images)
                self.images.append(image_matrix)
                self.labels.append([yaw, pitch])
                        
    def _augmentation(self):
        if (self.augmentation):
            crop = RandomCrop(self.outSize, self.C)
            flip = RandomHorizontalFlip(self.C)
            blur = GaussianBlur(self.C)
            sharp = RandomAdjustSharpness(self.C)
            mixup = SelfMixup(self.outSize, self.C)
            mixup_2 = SelfMixup_2(self.outSize, self.C)
            self.trsfrm = [crop, flip, blur, sharp, mixup, mixup_2]
        else:
            self.trsfrm = []

    def __getitem__(
            self,
            index: int):
        # refresh augmentation random parameter every image
        self._augmentation()
        randIdx = np.random.randint(len(self.trsfrm)+1)
        if self.augmentation and randIdx < len(self.trsfrm):
            image, yaw, pitch = self.trsfrm[randIdx]((self.images[index], 
                                                     self.labels[index][0], 
                                                     self.labels[index][1]))
            labels = [yaw, pitch]
            return image, labels
        else: 
            return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.images)