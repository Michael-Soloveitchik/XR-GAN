import sys
import os
import shutil
import cv2
from utils import remove_and_create, create_if_not_exists
from SubXR_configs_parser import SubXRParser

def initialze(configs):
    header = "Data"
    for data_type in configs[header]:
        for data_source in configs[header][data_type]:
            if configs[header][data_type][data_source]['in_dir']:
                remove_and_create(configs[header][data_type][data_source]['out_dir'])
                if data_source == "DRR":
                    for dir in os.listdir(configs[header][data_type][data_source]['in_dir']):
                        full_drr_path = os.path.join(configs[header][data_type][data_source]['in_dir'], dir)
                        for dir in ['X', 'Y','XY','YX']:
                            create_if_not_exists(os.path.join(full_drr_path, 'pre_DRR',dir))

    header = "Datasets"
    for data_type in configs[header]:
        for data_source in configs[header][data_type]:
            if configs[header][data_type][data_source]['in_dir']:
                remove_and_create(configs[header][data_type][data_source]['out_dir'])

if __name__ == '__main__':
    configs = SubXRParser()
    initialze(configs)


    #
    # for dir_path in os.listdir(DATA_apth):
    #     full_drr_path = os.path.join(DATA_apth, dir_path)
    #     for dir in ['X', 'Y','XY','YX']:
    #         create_if_not_exists(os.path.join(full_drr_path, 'pre_DRR',dir))

