import sys
import os
import shutil
import cv2

remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)
if __name__ == '__main__':
    DATA_apth  = 'C:\Michael research data\CT sagittal - Data'
    data_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Data'
    remove_and_create(data_path)
    remove_and_create(os.path.join(data_path, 'DRR'))
    remove_and_create(os.path.join(data_path, 'X-Ray'))
    for dir_path in os.listdir(DATA_apth):
        full_drr_path = os.path.join(DATA_apth, dir_path)
        for dir in ['X', 'Y','XY','YX']:
            create_if_not_exists(os.path.join(full_drr_path, 'pre_DRR',dir))

        remove_and_create(os.path.join(full_drr_path, 'DRR'))
        for dir in ['Input', 'Ulna','Radius']:
            remove_and_create(os.path.join(full_drr_path, 'DRR',dir))
            remove_and_create(os.path.join(data_path,'DRR',dir))
