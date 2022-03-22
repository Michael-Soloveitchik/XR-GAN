import numpy as np
import os
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import random
# import imgaug

XR_2_mask_AU = A.Compose([
    A.VerticalFlip(0.5),
    A.HorizontalFlip(0.5),
    A.ShiftScaleRotate(shift_limit=0.3,rotate_limit=360, p=1., border_mode=cv2.BORDER_CONSTANT,value=0),
    A.augmentations.transforms.PadIfNeeded(min_height=900, min_width=900),
    A.RandomCrop(900, 900, p=1.0),
    A.Sharpen(alpha=0.9,lightness=1.,p=0.2),
    A.augmentations.geometric.resize.RandomScale(scale_limit=0.3, interpolation=cv2.INTER_CUBIC, p=1.),
    A.augmentations.transforms.PadIfNeeded(min_height=300, min_width=300),
    A.CenterCrop(300, 300, p=1.0),

    # A.Emboss(),
    # A.RandomBrightnessContrast(),
    # A.Perspective (p=0.3),
    ])
no_crop_XR_AU = A.Compose([
    # A.Sharpen(alpha=0.9, lightness=1., p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=10, p=1., border_mode=cv2.BORDER_CONSTANT, value=0),
    # A.augmentations.transforms.PadIfNeeded(min_height=2187, min_width=2187),
    # A.CenterCrop(2187, 2187, p=1.0),
    # A.Emboss(),
    # A.RandomBrightnessContrast(),
    # A.Perspective (p=0.3),
])

sr_xr_complete_AU = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2,rotate_limit=180, p=1., border_mode=cv2.BORDER_CONSTANT,value=0),
    A.augmentations.geometric.resize.SmallestMaxSize(max_size=128,interpolation=cv2.INTER_CUBIC,p=1.),
    A.augmentations.geometric.resize.RandomScale(scale_limit=0.4, interpolation=cv2.INTER_CUBIC, p=1.),
    A.augmentations.transforms.PadIfNeeded(min_height=128, min_width=128),
    A.RandomCrop(128, 128, always_apply=True, p=1.0)
    # A.Emboss(),
    # A.RandomBrightnessContrast(),8
    # A.Perspective (p=0.3),
    ])

drr_complete_2_xr_complete_AU = A.Compose([
    A.augmentations.geometric.resize.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_CUBIC, p=1.),
    A.ShiftScaleRotate(shift_limit=0.5, rotate_limit=20, p=1.0),
    # A.Perspective (p=0.3),
    A.Sharpen(),
    A.VerticalFlip(0.5),
    A.RandomCrop(64, 64, always_apply=True, p=1.0)
    # A.Emboss(),
    # A.RandomBrightnessContrast(),
])
make_it_like_XR_AU = sr_xr_complete_AU
# make_it_like_xr_AU = sr_xr_complete_AU


A.Compose([

    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.3),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.2, rotate_limit=10, p=0.6, border_mode=cv2.BORDER_CONSTANT,value=0),
    A.OneOf([
        # A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        A.Perspective (p=0.3),
    ], p=0.4),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        # A.Emboss(),
        # A.RandomBrightnessContrast(),
    ], p=0.4),
    A.HueSaturationValue(p=0.3),
])

OLD = A.Compose([

    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.3),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.2, rotate_limit=15, p=0.6, border_mode=cv2.BORDER_CONSTANT,value=0),
    A.OneOf([
        # A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        A.Perspective (p=0.3),
    ], p=0.4),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen()
        # A.Emboss(),
        # A.RandomBrightnessContrast(),
    ], p=0.4),
    A.HueSaturationValue(p=0.3),
])


AUGMENTATIONS = {
    "sr_xr_complete_AU"             : sr_xr_complete_AU,
    "drr_complete_2_xr_complete_AU" : drr_complete_2_xr_complete_AU,
    "no_crop_XR_AU"                    : no_crop_XR_AU,
    "make_it_like_XR_AU": make_it_like_XR_AU,
    "XR_2_mask_AU" : XR_2_mask_AU
}

def parse_augmentation(augmentation):
    augmentation_k = augmentation
    if augmentation:
        if augmentation_k in AUGMENTATIONS:
            return lambda x: AUGMENTATIONS[augmentation_k](image=x)['image']
    return lambda x: x
