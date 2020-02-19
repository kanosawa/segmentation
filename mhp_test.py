import os
import glob
import numpy as np
from copy import deepcopy
import cv2

data_dir = '/root/dataset/LV-MHP-v2/train'
img_dir = os.path.join(data_dir, 'images')
label_dir = os.path.join(data_dir, 'parsing_annos')

img_name_list = glob.glob(os.path.join(img_dir, '*'))

for img_name in img_name_list:

    img = cv2.imread(img_name)

    base_name, _ = os.path.splitext(os.path.basename(img_name))
    label_name_list = glob.glob(os.path.join(label_dir, base_name + '_*'))

    label_sum = np.zeros_like(img)
    for label_name in label_name_list:

        label = cv2.imread(label_name)
        label[:, :, 0] = label[:, :, 2]
        label[:, :, 1] = label[:, :, 2]
        label_sum += label
    
    img[label_sum == 0] = 0    
    cv2.imshow('', img)
    cv2.waitKey()
    
    