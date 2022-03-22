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

def create_datasets(configs, dataset_type):
    for mode in ['train', 'test']:
        K = configs['Datasets'][dataset_type]['repeat_times_'+mode]
        in_dir_A_size = size_dir_content(configs['Datasets'][dataset_type]['in_dir_'+mode+'_A'])
        in_dir_B_size = size_dir_content(configs['Datasets'][dataset_type]['in_dir_'+mode+'_B'])
        seeds_permutations = np.random.permutation(max(in_dir_A_size,in_dir_B_size))*K
        if configs['Datasets'][dataset_type]['is_paired']:
            for side in configs['Datasets'][dataset_type]['out_sub_folders']:
                remove_and_create(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode,side))
                create_if_not_exists(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode,side))
            idx_im_name = 0
            j=0
            augmentation = parse_augmentation(configs['Datasets'][dataset_type]['augmentation_'+mode+'_'+side])
            transform_A = parse_transforms(configs['Datasets'][dataset_type]['transform_A'],dataset_type=dataset_type)
            transform_B = parse_transforms(configs['Datasets'][dataset_type]['transform_B'],dataset_type=dataset_type)
            for i, (im_A_name, im_B_name) in enumerate(tqdm(dirs_content([configs['Datasets'][dataset_type]['in_dir_'+mode+'_A'], configs['Datasets'][dataset_type]['in_dir_'+mode+'_B']], False))):
                im_A_raw = cv2.imread(os.path.join(configs['Datasets'][dataset_type]['in_dir_'+mode+'_A'], im_A_name))
                im_B_raw = cv2.imread(os.path.join(configs['Datasets'][dataset_type]['in_dir_'+mode+'_B'], im_B_name))
                im_A_raw_transformed = transform_A(im_A_raw, im_name=im_A_name)
                im_B_raw_transformed = transform_B(im_B_raw, im_name=im_B_name)
                seeds = np.arange(seeds_permutations[i], seeds_permutations[i] + K)
                # Augmentations
                for seed in seeds:
                    random.seed(seed);
                    np.random.seed(seed);
                    imgaug.random.seed(seed)
                    im_A_raw_transformed_augmented = augmentation(im_A_raw_transformed)
                    random.seed(seed);
                    np.random.seed(seed);
                    imgaug.random.seed(seed)
                    im_B_raw_transformed_augmented = augmentation(im_B_raw_transformed)
                    if (im_B_raw_transformed_augmented.sum() < (3000*255)):
                        if (j <(0.1*idx_im_name)):
                            j +=1
                            pass
                        else:
                            continue

                    im_A_transformed_augmented_name = '_'.join(im_A_name.split('_')[:-1])+'_{idx_im_name}.jpg'.format(idx_im_name=idx_im_name)
                    im_B_transformed_augmented_name = '_'.join(im_B_name.split('_')[:-1])+'_{idx_im_name}.jpg'.format(idx_im_name=idx_im_name)
                    cv2.imwrite(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, 'A', im_A_transformed_augmented_name), im_A_raw_transformed_augmented)
                    cv2.imwrite(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, 'B', im_B_transformed_augmented_name), im_B_raw_transformed_augmented)
                    idx_im_name+=1
        else:
            for side in configs['Datasets'][dataset_type]['out_sub_folders']:
                remove_and_create(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode,side))
                create_if_not_exists(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode,side))
                idx_im_name = 0
                augmentation = parse_augmentation(configs['Datasets'][dataset_type]['augmentation_'+mode+'_'+side])
                transform = parse_transforms(configs['Datasets'][dataset_type]['transform_' + side],dataset_type=dataset_type)
                for i, (im_name) in enumerate(tqdm(dirs_content([configs['Datasets'][dataset_type]['in_dir_'+mode+'_'+side]], False))):
                    im_raw = cv2.imread(os.path.join(configs['Datasets'][dataset_type]['in_dir_'+mode+'_'+side], im_name))
                    seeds = np.arange(seeds_permutations[i], seeds_permutations[i] + K)
                    im_raw_transformed = transform(im_raw, im_name=im_name)
                    # Augmentations
                    for seed in seeds:
                        random.seed(seed);
                        np.random.seed(seed);
                        imgaug.random.seed(seed)
                        im_raw_transformed_augmented = augmentation(im_raw_transformed)
                        im_transformed_augmented_name = '_'.join(im_name.split('_')[:-1])+'_{idx_im_name}.jpg'.format(idx_im_name=idx_im_name)
                        cv2.imwrite(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, side, im_transformed_augmented_name), im_raw_transformed_augmented)
                        idx_im_name+=1

if __name__ == '__main__':
    configs = SubXRParser()
    create_datasets(configs, "XR_complete_2_Ulna_and_Radius_Mask_complete")