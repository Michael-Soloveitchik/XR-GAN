import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import cv2
import torch
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'train', 'A')  # create a path '/path/to/data/trainA'
        self.dir_A_test = os.path.join(opt.dataroot,'test', 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot,'train', 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.input_classes, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.test_A_paths = sorted(make_dataset(self.dir_A_test, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = {clss: sorted(make_dataset(os.apth.join(self.dir_B,clss), opt.max_dataset_size)) for clss in opt.output_classes.split('_')}    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.test_A_size = len(self.test_A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        get_clossest_power_size = lambda t: int('1'+''.join(['0']*len(bin(t)[2:])),2) if '1' in bin(t)[3:] else t
        get_pad = lambda t : ((get_clossest_power_size(t)-t)//2, ((get_clossest_power_size(t)-t)-(get_clossest_power_size(t)-t)//2))
        self.transform_test_A = lambda x: np.transpose(np.pad((np.array(x.getdata())/(127.5)).astype(np.float32).reshape(x.size[1], x.size[0],3), (get_pad(x.size[1]), get_pad(x.size[0]), (0,0)), 'constant'), [2,0,1])-1. #get_transform(self.opt, grayscale=(input_nc == 1))
        self.opt = opt
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        test_A_paths = self.test_A_paths[index % self.test_A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B = {}
        B_path = {}
        for clss in self.opt.output_classes.split('_'):
            B_path[clss] = self.B_paths[clss][index_B]
            B_img = Image.open(B_path[clss]).convert('RGB')
            B[clss] = self.transform_B(B_img)[:1]
        A_img = Image.open(A_path).convert('RGB')
        test_A_img = Image.open(test_A_paths).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        test_A = self.transform_test_A(test_A_img)
        return {'A': A[:1], 'test_A':test_A[:1], 'A_paths': A_path, 'test_A_paths':test_A_paths}.update({'B_paths'+k:v for k,v in B_path.items()}).update({'B_'+k:v for k,v in B.items()}).update({'B':torch.cat([B[clss][None,...] for  clss in self.opt.output_classes.split('_')])})
    def collate(self, batch):
        elem = batch[0]
        a={k:(torch.stack([d[k]for d in batch]) if type(batch[0][k]) != str else np.stack([np.array(d[k])[None,...] for d in batch])) for k in elem if 'test' not in k}
        b={k:(torch.stack([torch.as_tensor(batch[0][k])]*len(batch)) if type(batch[0][k]) != str else np.stack([np.array(batch[0][k])[None,...]]*len(batch))) for k in elem if 'test' in k}
        a.update(b)
        return a
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size, self.test_A_size)
