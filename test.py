import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from Transforms.transforms import *
if 0 and __name__=='__main__':
    img = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Data\DRR_complete\Ulna_mask\Ulna_mask_00022.jpg")
    img3=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(plt.hist(img3.flatten()))
    plt.figure(3)
    plt.imshow(img3)

    img4 = cv2.threshold(img3, 229, 255,cv2.THRESH_BINARY)[1]
    print(plt.hist(img4.flatten()))
    plt.figure(4)
    plt.imshow(img4)
    # img,angle = self_rotate_transform(img)
    # img,center = self_crop_transform(img)
    # img = padding_transform(img)
    plt.imshow(img)
    plt.show()
if 0 and  __name__=='__main__':
    img = cv2.imread(r"G:\My Drive\CT-DRR\Data\raw X-Ray - Data\x rays drf\SynapseExport (5)\Image00002.jpg")
    plt.figure(0)
    plt.imshow(img)
    img,angle = self_rotate_transform(img)
    plt.figure(1)
    plt.imshow(img)
    img,center = self_crop_transform(img)
    plt.figure(2)
    plt.imshow(img)
    img = padding_transform(img)
    plt.imshow(img)
    plt.show()
if 0 and __name__ == '__main__':
    img1 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainA\Input_52.jpg")
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainA\Input_41.jpg")
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainB\xr_4.jpg")
    img3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainB\xr_774.jpg")
    img4=cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    for i, im in enumerate([img1,img2,img3,img4]):
        plt.figure(i)
        plt.hist(im.flatten())
        print(im.flatten().mean())
        print(im.flatten().var())
        print('___')
    plt.show()
    print('Shalom')

if 0 and __name__ == '__main__':
    p = r'/home/michael/PycharmProjects/XR-GAN/SAMPLEs/DRR_2_Ulana_and_Radius_Mask/web/images/epoch8800_fake_test_B.png'
    img = cv2.imread(p)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img)
    plt.show()
    print('shalom')

if 0 and __name__ == '__main__':
    p = r'/home/michael/Documents/final - CT/pre_DRR/Data'
    for f in os.listdir(p):
        if 'Mask' in f:
            a = f.split('Mask')
            a[0]=a[0].replace(' ','')
            n_f = 'Mask'.join(a)
            print(f,n_f)
            os.rename(os.path.join(p,f), os.path.join(p,n_f))
,
    "Datasets" : {
        "XR_complete_2_Mask_complete" :   {
            "model": "cycle_gan",
            "in_dir_train_A" : "./Data/DRR_complete/Hande",
            "in_dir_train_B" : [
                                "./Data/DRR_complete/Forehand_Mask",
                                "./Data/DRR_complete/Carpals_Mask",
                                "./Data/DRR_complete/Metacarpals_Mask",
                                "./Data/DRR_complete/ProximalPhalanges_Mask",
                                "./Data/DRR_complete/IntermediatePhalanges_Mask",
                                "./Data/DRR_complete/DistalPhalanges_Mask"
                                ]
            "in_dir_test_A" : "./Data/XR_complete/Hand",
            "in_dir_test_B" : "./Data/XR_complete/Hand",
            "out_dir" : "./Datasets/xr_complete2ulna_and_radius_mask_complete",
            "out_sub_folders" : [
                "A",
                "B"
            ],
             "transform_A": [
                             ],
             "transform_B": [
                                {
                                    "image_2_mask": [0.2,1.0]
                                }
                            ],
             "repeat_times_train" : 70,
             "repeat_times_test" : 1,
             "is_paired": true,
             "augmentation_train_A": "XR_2_mask_AU",
             "augmentation_train_B": "XR_2_mask_AU",
             "augmentation_test_A": "no_crop_XR_AU",
             "augmentation_test_B": "no_crop_XR_AU"
        },
        "DRR_complete_2_XR_complete" :   {
            "model": "cycle_gan",
            "in_dir_train_A" : "./Data/DRR_complete/Input",
            "in_dir_train_B" : "./Data/XR_complete/XR",
            "in_dir_test_A" : "./Data/DRR_complete/Input",
            "in_dir_test_B" : "./Data/XR_complete/XR",
            "out_dir" : "./Datasets/xr_complete2ulna_and_radius_mask_complete",
            "out_sub_folders" : [
                "A",
                "B"
            ],
             "transform_A": [
                             ],
             "transform_B": [
                                {
                                    "intersect_with_mask": [
                                                            "cycle_gan_classic",
                                                            "/home/michael/PycharmProjects/XR-GAN/SAMPLEs/XR_2_Ulana_and_Radius_Mask",
                                                             10000
                                                           ]
                                }
                            ],
             "repeat_times_train" : 10,
             "repeat_times_test" : 5,
             "is_paired": false,
             "augmentation_train_A": "drr_complete_2_xr_complete_AU",
             "augmentation_train_B": "drr_complete_2_xr_complete_AU",
             "augmentation_test_A": "drr_complete_2_xr_complete_AU",
             "augmentation_test_B": "drr_complete_2_xr_complete_AU",
             "train_pass_condition": null
        }
    },