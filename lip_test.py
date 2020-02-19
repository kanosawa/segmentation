import os
import glob
import cv2

data_dir = '/root/dataset/LIP'
img_dir = os.path.join(data_dir, 'TrainVal_images/TrainVal_images/train_images')
label_dir = os.path.join(data_dir, 'TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations')

img_name_list = glob.glob(os.path.join(img_dir, '*'))

for img_name in img_name_list:

    base_name, _ = os.path.splitext(os.path.basename(img_name))
    label_name = os.path.join(label_dir, base_name + '.png')

    img = cv2.imread(img_name)
    label = cv2.imread(label_name)

    img[label == 0] = 0

    cv2.imshow('', img)
    cv2.waitKey()
    # cv2.imshow('', label)
    # cv2.waitKey()
