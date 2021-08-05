import numpy as np
import os
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import random
import imgaug
win_size =512
def Augment_DRR(drr_path, file_name, j,seed_n):

    im = cv2.imread(os.path.join(drr_path,file_name))
    random.seed(seed_n)
    np.random.seed(seed_n)
    imgaug.random.seed(seed_n)
    for AU in [Au1]:

        augmented_im = AU(image=im)['image'][130:130+win_size, 250:250+win_size]
        plt.imshow(augmented_im[130:130+win_size, 250:250+win_size])
        new_filename = ("{0:2d}".format(j) + file_name).replace(' ', '0')
        im = cv2.imwrite(os.path.join(drr_path, new_filename),augmented_im)
            # plt.show()



Au1 = A.Compose([

        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.2, rotate_limit=15, p=0.6),
        A.OneOf([
            # A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.Perspective (p=0.3),
        ], p=0.4),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.4),
        A.HueSaturationValue(p=0.3),
    ])