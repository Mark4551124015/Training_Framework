import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import pandas as pd
import numpy as np
from glob import glob
import time
import random
import sys
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
from sklearn.model_selection import train_test_split
import shutil

origin_path_3M = r'/data/data/student/fengzexin/datasets/RETFOUND_OCT/OCTA-500/unziped/OCTA-500_ground_truth/OCTA_3M/Projection Maps/OCTA(ILM_OPL)/*.bmp'
origin_path_6M = r'/data/data/student/fengzexin/datasets/RETFOUND_OCT/OCTA-500/unziped/OCTA-500_ground_truth/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/*.bmp'
mask_path_3M =r'/data/data/student/fengzexin/datasets/RETFOUND_OCT/OCTA-500/unziped/OCTA-500_ground_truth/OCTA_3M/GroundTruth/*.bmp'
mask_path_6M =r'/data/data/student/fengzexin/datasets/RETFOUND_OCT/OCTA-500/unziped/OCTA-500_ground_truth/OCTA_6M/GroundTruth/*.bmp'
origin_csv_path = r'/data/data/student/fengzexin/datasets/OCTA_500/lable.csv'

csv_path = r'./datasets/octa_500/'

class DatasetGenerator(Dataset):
    def __init__(self, csv_path=origin_csv_path):
        """
        Args: 
        path_to_img_dir: path to image and mask directory.
        data_type: ['ROSE-1/DVC/', 'ROSE-1/SVC/', 'ROSE-1/SVC_DVC/','ROSE-2/']
        """
        imageIDs, maskIDs = [], []
        with open(csv_path, 'r')as f:
            f = f.readlines()
            for line in f:
                octa, mask = line.strip().split(' ')
                imageIDs.append(octa)
                maskIDs.append(mask)
        self.imageIDs = imageIDs
        self.maskIDs = maskIDs
        
        self.transform_octa = transforms.Compose([
            transforms.Resize(400),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its masks
        """
        image = self.imageIDs[index]
        name_img = image.split("/")[-1]
        mask = self.maskIDs[index]
        name_mask = image.split("/")[-1]


        image = Image.open(image).convert('L')
        # image = Image.open(image)
        origin = self.transform_octa(image)
        mask = cv2.imread(mask, cv2.COLOR_BGR2GRAY)
        mask = np.where(mask >= 128, 1, 0)
        mask = np.int8(mask)
        mask = Image.fromarray(mask)
        mask = self.transform_octa(mask)

        return origin, mask

    def __len__(self):
        return len(self.imageIDs)

    def getFileName(self):
        return self.name


def get_sets():
    train_set = DatasetGenerator(csv_path+'train.csv')
    val_set = DatasetGenerator(csv_path+'val.csv')
    test_set = DatasetGenerator(csv_path+'test.csv')
    return train_set, val_set, test_set


if __name__ == "__main__":
    Path = './datasets/octa_500/{}.csv'
    image_format = '/data/data/student/fengzexin/datasets/OCTA_500/bmp/{}.bmp /data/data/student/fengzexin/datasets/OCTA_500/masks/{}.bmp'
    set = 'train'
    with open(Path.format(set), 'w') as f:
        for i in range(240):
            no = str(i+10001)
            f.write(image_format.format(no, no)+'\n')
    set = 'val'
    with open(Path.format(set), 'w') as f:
        for i in range(10):
            no = str(i+10241)
            f.write(image_format.format(no, no)+'\n')
    set = 'test'
    with open(Path.format(set), 'w') as f:
        for i in range(50):
            no = str(i+10251)
            f.write(image_format.format(no, no)+'\n')