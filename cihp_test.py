import os
import glob
import numpy as np
from copy import deepcopy
import cv2

data_dir = '/root/dataset/CIHP/instance-level_human_parsing/Training'
img_dir = os.path.join(data_dir, 'Images')
label_dir = os.path.join(data_dir, 'Category_ids')
human_id_dir = os.path.join(data_dir, 'Human_ids')

img_name_list = glob.glob(os.path.join(img_dir, '*'))

for img_name in img_name_list:

    base_name, _ = os.path.splitext(os.path.basename(img_name))
    label_name = os.path.join(label_dir, base_name + '.png')
    human_id_name = os.path.join(human_id_dir, base_name + '.png')

    img = cv2.imread(img_name)
    label = cv2.imread(label_name)
    human_id = cv2.imread(human_id_name)

    human_id_max = np.max(human_id)

    for max_id in range(1, human_id_max + 1):
        img_tmp = deepcopy(img)
        img_tmp[human_id != max_id] = 0

        cv2.imshow('', img_tmp)
        cv2.waitKey()
    # cv2.imshow('', label)
    # cv2.waitKey()
