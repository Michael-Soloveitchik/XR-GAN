import sys
import os
import shutil

from tqdm import tqdm
from SubXR_configs_parser import SubXRParser
import cv2
from utils import *
import numpy as np
import utils
from Augmentations.augmentations import *
from Transforms.transforms import *
import imgaug
import subprocess

def train_models(configs, model_name):
    cmd = ''
    for k,v in configs['Models'][model_name]['parametrs'].items():
        cmd += (' --'+k+(' '+str(v) if v is not None else ""))
    cmd = "cd /home/michael/PycharmProjects/XR-GAN; "+\
          "conda activate SubXR-GAN; "+\
           cmd[3:]
    print(cmd)
    subprocess.call(cmd)
if __name__ == '__main__':
    configs = SubXRParser()
    train_models(configs, "XR_2_Ulana_and_Radius_Mask")

