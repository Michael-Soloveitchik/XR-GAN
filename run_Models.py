import sys
import os
import shutil
import torch
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from Models.Pix2Pix_single.models.networks import define_G
remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)
if __name__ == '__main__':

    # DATA_apth  = ' '.join(sys.argv[1:])
    # cmd1 = \
    #python ./Models/Pix2Pix-single/train.py --continue_train --epoch_count 6 --load_iter 2276 --checkpoints_dir ./Models/ --gan_mode lsgan --dataroot ./Datasets/xr2radius --name xr2radius --model pix2pix --batch_size 1 --save_by_iter --save_latest_freq 100 --save_epoch_freq 1 --continue_train --dataset_mode unaligned --direction AtoB --crop_size 512 --load_size 512 --no_flip --serial_batches --preprocess None
    #python ./Models/Pix2Pix-single/train.py --checkpoints_dir ./Models/ --gan_mode lsgan --dataroot ./Datasets/xr2radius --name xr2radius --model pix2pix --batch_size 1 --save_by_iter --save_latest_freq 100 --save_epoch_freq 1 --dataset_mode unaligned --direction AtoB --crop_size 512 --load_size 512 --no_flip --serial_batches --preprocess None
    model_name = r"C:\Users\micha\PycharmProjects\CT_DRR\Models\xr2radius\train_opt.txt"
    with open(model_name, 'r') as f:
        opt = f.readlines()[1:-1]
        opt = [l.strip().split(':') for l in opt]
        opt = [[l[0],l[1][1:].split(' ')[0]] for l in opt]
        options = ''.join([' --'+a[0]+' '+a[1] for a in opt if ((a[0] not in ['max_dataset_size', 'suffix','isTrain',' False',' True']) and (a[1] not in ['True','False']))])
    sys.argv = ('train.py'+''.join(options)).split(' ')
    print(sys.argv)

    opt = TrainOptions().parse()  # get test options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    model = torch.hub('junyanz/pytorch-CycleGAN-and-pix2pix')
