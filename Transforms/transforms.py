import numpy as np
import os
import shutil
import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.hub
# from Augmentations.augmentations import Augment_DRR
from numba import prange
# import imgaug
import matplotlib.pyplot as plt
from Models.sr_gan.model import Generator
import cv2
import subprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
LR = 400
HR = 800
TR = 700
TEST = 1.1
SR_GAN = Generator().to(device)

def padding_transform(im, W_pad=70, H_pad=70, dataset_type="", im_name=""):
    new_H = max((H_pad - im.shape[0])//2, 0)
    new_W = max((W_pad - im.shape[1])//2, 0)
    return np.pad(im,[(new_H,),(new_W,),(0,)],mode='constant')

def self_crop_transform(im):
    def get_circles_of_image(im):
        # gray-scalling
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        output = im + 0
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=15, param2=5, minRadius=20, maxRadius=70)
        center = im.shape[0]//2, im.shape[1]//2
        r  = None
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                center = (max(int(x-3*r), 0),int(y))
                break
            # show the output image
            plt.imshow(output)
            plt.show()#np.hstack([im, output]))
        return center, r#np.median(Pangles)
    center, r = get_circles_of_image(im)
    if r is None:
        w = center[0]
        h = center[1]
    else:
        w1, w2 = min(int(r*4),center[0]), min(int(r*4),im.shape[0]-center[0])
        h1, h2 = min(int(r*2),center[1]), min(int(r*2),im.shape[1]-center[1])
        # h = min(min(center[1],480)/1.5, min(im.shape[0]-center[1],320))*2.5
    # w=min(center[0],800)
    # LR = int(min(h,w))
    # print(h1,h2,w1,w2)
    # print(center[1]-h1,center[1]+h2,center[0]-w1,center[0]+w2)
    im = im[center[1]-h1:center[1]+h2,center[0]-w1:center[0]+w2]
    return im, center

def self_rotate_transform(im, dataset_type="", im_name=""):
    def get_orientation_of_image(im):
        # gray-scalling
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # bluring
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # Canny edges
        low_threshold = 10
        high_threshold =100
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        # compute orientation
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 170  # minimum number of pixels making up a line
        max_line_gap = 30  # maximum gap in pixels between connectable line segments

        line_image = np.copy(im) * 0  # creating a blank to draw lines on
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        Plines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
        Pangles = []
        Pdists = []
        if Plines is None:
            return 0
        for line in Plines:
            for x1,y1,x2,y2 in line:
                if x1>x2:
                    tx,ty=x1+0,y1+0
                    x1,y1=x2+0,y2+0
                    x2,y2=tx+0,ty+0
                if (x2-x1)>1e-05:
                    Pangles.append(np.arctan((y2-y1)/(x2-x1))/np.pi*180)
                else:
                    Pangles.append(0)
                # Pdists.append(np.sqrt((x1-x2)**2+(y1-y2)**2))
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            # Draw the lines on the  image
        # lines_edges = cv2.addWeighted(im, 0.8, line_image, 1, 0)
        # plt.imshow(lines_edges)
        # plt.show()
        # plt.hist(Pangles)
        # plt.show()
        return np.median(Pangles)

    angle, new_angle = 0,1
    cum_angle=0
    sh = im.shape
    if sh[0] > sh[1]:
        im = flip_rotate_90_counter_clockwisw(im, None, None)
    rot_im = im*1.
    while np.abs(angle-new_angle)> 1e-01:
        angle = get_orientation_of_image(rot_im)
        rot_im = rotate_transform(rot_im, angle, None,None)
        cum_angle+=angle
        new_angle = get_orientation_of_image(rot_im)
    im = rotate_transform(im, cum_angle, None,None)
    plt.imshow(im)
    plt.show()
    return im, cum_angle

def image_2_mask_transform(im, l_tresh,h_tresh, dataset_type, im_name):
    im_cp = im/255.
    mask = (l_tresh <= im_cp) & (im_cp <= h_tresh)
    im[mask] = 255
    im[~mask] = 0
    return im
def flip_rotate_90_counter_clockwisw(im, dataset_type, im_name):
    im=cv2.flip(im,flipCode=1)
    im=cv2.transpose(im)
    return im
def rotate_transform(nh_i_im, angle, dataset_type, im_name):
    nh_i_im_g = cv2.cvtColor(nh_i_im, cv2.COLOR_BGR2GRAY) if len(nh_i_im.shape) > 2 else nh_i_im
    if np.abs(np.abs(angle)) < 45:
        # nh_i_im_g = flip_rotate_90_counter_clockwisw(nh_i_im_g,dataset_type, im_name)
        angle = np.sign(angle)*np.abs(angle)
        # print(angle)
    # Detect keypoints (features) cand calculate the descriptors
    cy, cx = nh_i_im_g.shape
    center = (cx // 2, cy // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    h_i_im_g = cv2.warpAffine(src=nh_i_im_g, M=M, dsize=(nh_i_im_g.shape[1], nh_i_im_g.shape[0]))
    h_i_im_g_bgr = cv2.cvtColor(h_i_im_g, cv2.COLOR_GRAY2BGR)
    return h_i_im_g_bgr
def crop_transform(im, H=800, W=None, dataset_type="", im_name=""):
    if W is None:
        W = H + 0
    im_c_h, im_c_w = im.shape[0]//2, im.shape[1]//2
    im = im[max(im_c_h-H//2,0):max(im_c_h+H//2,0), max(im_c_w-W//2,0):max(im_c_w+W//2,0)]
    im = padding_transform(im, H,W)
    return im
def translate_transform(im, dx, dy, dataset_type, im_name):
    im = np.roll(im, dy, axis=0)
    im = np.roll(im, dx, axis=1)
    if dy>0:
        im[:dy, :] = 0
    elif dy<0:
        im[dy:, :] = 0
    if dx>0:
        im[:, :dx] = 0
    elif dx<0:
        im[:, dx:] = 0
    return im

def highpass_transform(im, dataset_type, im_name):
    up_down_im = cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(im))))
    highpasss_im = im-up_down_im
    # highpasss_im = (highpasss_im / highpasss_im.max())*255.
    return highpasss_im

def resize_transform(im, H,W, dataset_type, im_name):
    LR_im = cv2.resize(im, (H, W))
    return LR_im
def random_cut_and_crop(im, H, size, dataset_type, im_name):
    x,y = np.random.rantin(0,im.shape[0]-H,size), np.random.rantin(0,im.shape[0]-H,size)
    patches = []
    for x_i, y_i in zip(x,y):
        patches.append(im[x_i:x_i+H,y_i:y_i+H])
    return patches

def super_resolution_transform(im, SR_GAN_path, LR=384, HR=768, TR=650, dataset_type="", im_name=""):
    SR_GAN.load_state_dict(torch.load(SR_GAN_path, map_location=device))
    SR_GAN.eval()
    # SR_GAN.half()
    if im.shape[1]<TR:
        LR_im = resize_transform(im, LR,LR,dataset_type, im_name)
        torch_LR_im = (torch.from_numpy(np.transpose(LR_im, [2, 0, 1])[None]) / 255.).to(device)
        torch_HR_im = SR_GAN(torch_LR_im)
        HR_im = np.transpose(torch_HR_im[0].float().cpu().detach().numpy(), [1, 2, 0])*255
    else:
        HR_im = resize_transform(im, HR,HR,dataset_type, im_name)
    return HR_im
def binarization_transform(im, tresh=229,dataset_type="", im_name=""):
    binarized_im = cv2.threshold(im, tresh, 255,cv2.THRESH_BINARY)[1]/255
    return binarized_im
def intersect_with_mask(im,model_name,checkpoints_dir,iter):
    pass

def get_mask_for_image(im,model_name,checkpoints_dir,iter):
    temp_drr2xr_dir = os.path.abspath(f"./Datasets/temp")
    checkpoints_dir = os.path.abspath(checkpoints_dir)
    create_if_not_exists(temp_drr2xr_dir)
    cv2.imwrite(os.path.join(temp_drr2xr_dir,'drr2xr.jpg'),im)

    # Cycle-GAN activation
    cmd = (f"python ./Models/{model_name}/test.py "+\
            f"--model pix2pix"+\
            f"--results_dir {temp_drr2xr_dir}"
            f"--dataroot {temp_drr2xr_dir} "+\
            f"--load_iter {iter} "+\
            f"--netG unet_256 "+\
            f"--checkpoints_dir {checkpoints_dir} "+\
            f"--phase test "+\
            f"--no_dropout")
    subprocess.call(cmd)
    xr_im = cv2.cvtColor(cv2.imread(os.path.join(temp_drr2xr_dir,f'test_latest_iter{iter}','images','drr2xr_fake.png')), cv2.COLOR_BGR2RGB)
    shutil.rmtree(temp_drr2xr_dir)
    return xr_im

def drr_2_xr_style_transform(im, model_dir, iter, mask_name, dataset_type, im_name):
    dataset_name = dataset_type.lower()
    temp_drr2xr_dir = os.path.abspath("./Datasets/{name}/temp".format(name=dataset_name))
    create_if_not_exists(temp_drr2xr_dir)
    cv2.imwrite(os.path.join(temp_drr2xr_dir,'drr2xr.jpg'),im)

    # Cycle-GAN activation
    shape_inp = im.shape[0]
    style_transform_name = model_dir.split('\\')[-1]
    model_dir = '\\'.join(model_dir.split('\\')[:-2])
    cmd = (f"python {model_dir}\\test.py --model_suffix _A --dataroot {temp_drr2xr_dir} --name . --load_iter {iter} --netG unet_16 --netG_hp resnet_1blocks --netG_lp resnet_1blocks --checkpoints_dir {os.path.join(model_dir, 'SAMPLES',style_transform_name)} --model test --no_dropout --crop_size {shape_inp} --results_dir {temp_drr2xr_dir}")
    subprocess.call(cmd)
    xr_im = cv2.imread(os.path.join(temp_drr2xr_dir,f'test_latest_iter{iter}','images','drr2xr_fake.png'))
    shutil.rmtree(temp_drr2xr_dir)
    return xr_im

TRANSFORMS = {
    "down_sample":      resize_transform,
    "crop":             crop_transform,
    "self_crop":        self_crop_transform,
    "translate":        translate_transform,
    "up_sample":        resize_transform,
    "SR_GAN":           super_resolution_transform,
    "DRR_2_XR" :        drr_2_xr_style_transform,
    "subtraction_AU":   super_resolution_transform,
    "binarization":     binarization_transform,
    "random_cut_and_crop": random_cut_and_crop,
    "highpass":        highpass_transform,
    'image_2_mask':    image_2_mask_transform
}



def parse_transforms(transforms, dataset_type):
    transforms_functions = []
    if transforms:
        for transform_i in transforms:
            transform_k, transform_args = list(transform_i.items())[0]
            print(transform_k)
            print(TRANSFORMS[transform_k])
            if transform_k in TRANSFORMS:
                transforms_functions.append((TRANSFORMS[transform_k], transform_args+[dataset_type]))
    def transform(x,im_name):
        result = x
        for (transform_function, transform_args) in transforms_functions:
            result = transform_function(result, *(transform_args+[im_name]))
        return result
    return transform