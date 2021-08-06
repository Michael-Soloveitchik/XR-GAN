import sys
import os
import shutil

import numpy as np
import torch
import os
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import cv2

from Models.Pix2Pix_single.models.networks import define_G
remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)

# XR_images_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets\xr_real'
XR_images_path  = r'./Datasets/xr_real'
if __name__ == '__main__':

    DATA_apth  = ' '.join(sys.argv[1:])
    # cmd1 = \
    #python ./Models/Pix2Pix-single/train.py --continue_train --epoch_count 6 --load_iter 2276 --checkpoints_dir ./Models/ --gan_mode lsgan --dataroot ./Datasets/xr2radius --name xr2radius --model pix2pix --batch_size 1 --save_by_iter --save_latest_freq 100 --save_epoch_freq 1 --continue_train --dataset_mode unaligned --direction AtoB --crop_size 512 --load_size 512 --no_flip --serial_batches --preprocess None
    #python ./Models/Pix2Pix-single/train.py --checkpoints_dir ./Models/ --gan_mode lsgan --dataroot ./Datasets/xr2radius --name xr2radius --model pix2pix --batch_size 1 --save_by_iter --save_latest_freq 100 --save_epoch_freq 1 --dataset_mode unaligned --direction AtoB --crop_size 512 --load_size 512 --no_flip --serial_batches --preprocess None
    for model_name in os.listdir(DATA_apth):
        # if model_name.startswith('Pix2Pix'):
        #     continue
        # try:
            not_relevant_flags = ['phase','dataset_mode','dataroot','continue_train','isTrain','checkpoints_dir', 'max_dataset_size', 'suffix','serial_batches']#,'beta1','display_env','display_freq','display_id','display_ncols','display_port','display_server','epoch_count','gan_mode lsgan','lambda_L1','lr','lr_decay_iters','lr_policy linear','n_epochs','n_epochs_decay','pool_size','print_freq','save_epoch_freq','save_latest_freq', 'update_html_freq', 'continue_train']
            relevant_flags = ['name','model','netG','load_size','norm','ngf','no_flip','no_dropout','no_html']
            runs_path  = os.path.join(DATA_apth, model_name,'runs')
            if not os.path.exists(runs_path):
                continue
            for run in os.listdir(runs_path):
                train_opt_path = os.path.join(runs_path,run,'train_opt.txt')#r"C:\Users\micha\PycharmProjects\CT_DRR\Models\xr2radius\train_opt.txt"
                with open(train_opt_path, 'r') as f:
                    options = f.readlines()[1:-1]
                    options = [l.strip().split(':') for l in options]
                    options = [[l[0],l[1][1:].split(' ')[0]] for l in options]
                    options = ''.join([' --'+a[0]+' '+a[1] for a in options if (not (a[0] in not_relevant_flags) and (a[1] not in ['True','False']))])
                    sys.argv = ('train.py'+''.join(options)+' --checkpoints_dir ./Models/'+ model_name+'/runs'+\
                                                            ' --phase test'+\
                                                            ' --dataroot '+XR_images_path+\
                                                            ' --dataset_mode single').split(' ')
                    # print(sys.argv)

                    opt = TrainOptions().parse()  # get test options
                    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
                    model = create_model(opt)
                    model.setup(opt)
                    G = model.netG
                    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # G.to(device)
                    images = torch.tensor(np.concatenate([torch.tensor((np.transpose(cv2.imread(os.path.join(XR_images_path, f)), [2, 0, 1]).astype(np.float32)[
                                                np.newaxis, ...] / 256) - 0.5) for f in os.listdir(XR_images_path)],0))
                    processed_images = G(images)
