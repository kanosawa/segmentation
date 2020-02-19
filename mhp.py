import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class MHP(Dataset):

    def __init__(self, data_dir, n_classes):

        label_dir = os.path.join(data_dir, 'parsing_annos')
        self.label_name_list = glob.glob(os.path.join(label_dir, '*'))
        self.img_dir = os.path.join(data_dir, 'images')
        self.n_classes = n_classes

    def __len__(self):
        return len(self.label_name_list)
        #return 100
        #return 1

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        label_name = self.label_name_list[idx]
        base_name, _ = os.path.splitext(os.path.basename(label_name))
        base_name_splited = base_name.split('_')

        img_name = os.path.join(self.img_dir, base_name_splited[0] + '.jpg')
        img = cv2.imread(img_name)

        label = cv2.imread(label_name)
        label[:, :, 0] = label[:, :, 2]
        label[:, :, 1] = label[:, :, 2]
        img[label == 0] = 0

        label_x = np.sum(label, axis=0)[:, 0]
        x_where = np.where(label_x > 0)[0]
        x_min = x_where[0]
        x_max = x_where[-1]

        label_y = np.sum(label, axis=1)[:, 0]
        y_where = np.where(label_y > 0)[0]
        y_min = y_where[0]
        y_max = y_where[-1]

        img = img[y_min : y_max + 1, x_min : x_max + 1]
        label = label[y_min : y_max + 1, x_min : x_max + 1]

        img = cv2.resize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        img = (img / 255).astype('float32')

        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        label = label[:, :, 0].astype('int64')

        return img, label
    