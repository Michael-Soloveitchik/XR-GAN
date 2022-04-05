import sys
import os
import shutil

import numba
from tqdm import tqdm
from SubXR_configs_parser import SubXRParser
import cv2
from utils import *
import numpy as np
import utils
from Augmentations.augmentations import *
from Transforms.transforms import *
import imgaug
def skip_im(im, j1,j2,j3,n):
    if (im.sum() < (5000 * 100)):
        if (im.sum() < (3000 * 100)):
            if (im.sum() < (1000 * 100)):
                if (j1 < (0.07 * n)):
                    j1 += 1
                    # print(j1,idx_im_name)
                    return True, j1,j2,j3
                else:
                    return False, j1,j2,j3
            else:
                if (j2 < (0.1 * n)):
                    j2 += 1
                    # print(j2,idx_im_name)
                    return True, j1,j2,j3
                else:
                    return False, j1,j2,j3
        else:
            if (j3 < (0.1 * n)):
                j3 += 1
                # print(j3, idx_im_name)
                return True, j1,j2,j3
            else:
                return False, j1,j2,j3
    else:
        return True, j1, j2, j3

def create_datasets(configs, dataset_type):
    # Alternating over 'train' and 'test'
    for mode in ['train', 'test']:
        # configing K - the parameter of repeations
        K = configs['Datasets'][dataset_type]['repeat_times_'+mode]
        # for a in dirs_content(configs['Datasets'][dataset_type]['in_dir_' + mode + '_A'], random=False):
        #     print(a)
        in_dir_A_size = max([size_dir_content(a) for a in configs['Datasets'][dataset_type]['in_dir_'+mode+'_A']])
        in_dir_B_size = max([size_dir_content(a) for a in configs['Datasets'][dataset_type]['in_dir_'+mode+'_B']])
        # creating seeds
        seeds_permutations = np.random.permutation(max(in_dir_A_size,in_dir_B_size))*K
        if configs['Datasets'][dataset_type]['is_paired']:
            for side in configs['Datasets'][dataset_type]['out_sub_folders']:
                remove_and_create(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode,side))
                create_if_not_exists(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode,side))
            idx_im_name = 0
            j1,j2,j3=0,0,0
            augmentation = parse_augmentation(configs['Datasets'][dataset_type]['augmentation_'+mode+'_'+side])
            transform_A = parse_transforms(configs['Datasets'][dataset_type]['transform_A'],dataset_type=dataset_type)
            transform_B = parse_transforms(configs['Datasets'][dataset_type]['transform_B'],dataset_type=dataset_type)
            dir_names = [a for a in dirs_content([*configs['Datasets'][dataset_type]['in_dir_'+mode+'_A'], *configs['Datasets'][dataset_type]['in_dir_'+mode+'_B']], False)]
            for i in tqdm(numba.prange(len(dir_names))):
                im_names = dir_names[i]
                im_A_name = im_names[0]
                im_B_names = im_names[1:]
                im_A_raw = cv2.imread(os.path.join(configs['Datasets'][dataset_type]['in_dir_' + mode + '_A'][0], im_A_name))
                im_A_raw_transformed = transform_A(im_A_raw, im_name=im_A_name)
                seeds = np.arange(seeds_permutations[i], seeds_permutations[i] + K)

                dir_A = '_'.join(im_A_name.split('_')[:-1])
                create_if_not_exists(os.path.abspath(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, 'A', dir_A)))
                # Augmentations
                for seed in seeds:
                    random.seed(seed);
                    np.random.seed(seed);
                    imgaug.random.seed(seed)
                    im_A_raw_transformed_augmented = augmentation(im_A_raw_transformed)

                    keep,j1,j2,j3 = skip_im(im_A_raw_transformed_augmented,j1,j2,j3,idx_im_name)
                    if not keep:
                        continue
                    for i_B, im_B_name in enumerate(im_B_names):
                        dir_B = '_'.join(im_B_name.split('_')[:-1])
                        create_if_not_exists(os.path.abspath(
                            os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, 'B', dir_B)))

                        im_B_raw = cv2.imread(
                            os.path.join(configs['Datasets'][dataset_type]['in_dir_' + mode + '_B'][i_B],
                                         im_B_name))
                        im_B_raw_transformed = transform_B(im_B_raw, im_name=im_B_name)

                        im_B_raw_transformed_augmented = augmentation(im_B_raw_transformed)

                        cv2.imwrite(os.path.abspath(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, 'B', dir_B, str(idx_im_name)+'.jpg')), im_B_raw_transformed_augmented)
                    cv2.imwrite(os.path.abspath(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, 'A', dir_A,
                                                             str(idx_im_name)+'.jpg')), im_A_raw_transformed_augmented)
                    # print(os.path.abspath(os.path.join(configs['Datasets'][dataset_type]['out_dir'], mode, 'A',dir_A, str(idx_im_name)+'.jpg')))
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
    create_datasets(configs, "XR_complete_2_Mask_complete")