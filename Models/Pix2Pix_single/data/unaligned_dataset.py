import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


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
        self.dir_trainA = os.path.join(opt.dataroot, opt.name, 'trainA')  # create a path '/path/to/data/trainA'
        self.dir_trainB = os.path.join(opt.dataroot, opt.name, 'trainB')  # create a path '/path/to/data/trainB'
        self.dir_testA = os.path.join(opt.dataroot, opt.name, 'testA')  # create a path '/path/to/data/testA'
        self.dir_testB = os.path.join(opt.dataroot, opt.name, 'testB')  # create a path '/path/to/data/testB'
        self.dir_XR = os.path.join(opt.dataroot, 'xr_real')  # create a path '/path/to/data/xr_real'

        self.trainA_paths = sorted(make_dataset(self.dir_trainA, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.trainB_paths = sorted(make_dataset(self.dir_trainB, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.testA_paths = sorted(make_dataset(self.dir_testA, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.testB_paths = sorted(make_dataset(self.dir_testB, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.XR_paths = sorted(make_dataset(self.dir_XR, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.trainA_size = len(self.trainA_paths)  # get the size of train dataset A
        self.trainB_size = len(self.trainB_paths)  # get the size of train dataset B
        self.testA_size = len(self.testA_paths)  # get the size of test dataset A
        self.testB_size = len(self.testB_paths)  # get the size of test dataset B
        self.XR_size = len(self.XR_paths)  # get the size of dataset B
        if self.trainA_size != self.trainB_size:
            print("The size of trainA != trainB.")
            exit(0)

        if self.testA_size != self.testB_size:
            print("The size of testA != testB.")
            exit(0)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        trainA_path = self.trainA_paths[index % self.trainA_size]  # make sure index is within then range
        trainB_path = self.trainB_paths[index % self.trainB_size]
        testA_path = self.testA_paths[index % self.testA_size]  # make sure index is within then range
        testB_path = self.testB_paths[index % self.testB_size]
        XR_path = self.XR_paths[index % self.XR_size]

        trainA_img = Image.open(trainA_path).convert('RGB')
        trainB_img = Image.open(trainB_path).convert('RGB')
        testA_img = Image.open(testA_path).convert('RGB')
        testB_img = Image.open(testB_path).convert('RGB')
        XR_img = Image.open(XR_path).convert('RGB')
        # apply image transformation
        trainA = self.transform_A(trainA_img)
        trainB = self.transform_B(trainB_img)
        testA = self.transform_A(testA_img)
        testB = self.transform_B(testB_img)
        XR_img = self.transform_A(XR_img)

        return {'trainA': trainA, 'trainB': trainB, 'testA': testA, 'testB': testB, 'real_XR':XR_img,
                'trainA_paths': trainA_path, 'trainB_paths': trainB_path, 'testA_paths': trainA_path, 'testB_paths': trainB_path, 'real_XR_paths': XR_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.trainA_size, self.trainB_size)
