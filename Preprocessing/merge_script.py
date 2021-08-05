import os
import shutil
from Preprocessing.rotate import regularize_orientation

def merge_folders(DATA_apth):
    totall = 0
    for dir_path in os.listdir(DATA_apth):
        dataset_path = os.path.join(r'C:\Users\micha\PycharmProjects\CT_DRR\Data','DRR')
        DRR_path = os.path.join(DATA_apth, dir_path,'DRR')
        pre_DRR_path = os.path.join(DATA_apth, dir_path,'pre_DRR')
        for s in ['Input', 'Ulna', 'Radius']:
            in_dir_totall = 0

            for i, suf in enumerate(['X', 'Y', 'XY', 'YX']):
                n = len([t for t in os.listdir(os.path.join(pre_DRR_path, suf)) if t.endswith('.png') and t.startswith(s)])
                j = 0
                for f in sorted(os.listdir(os.path.join(pre_DRR_path, suf))):

                    if f.endswith('.png') and f.startswith(s):

                        new_f = ("{0:3d}".format(i * n + j) + '.png').replace(' ', '0')
                        new_totall = ("{0:5d}".format(totall+i * n + j) + '.png').replace(' ', '0')
                        shutil.copy(os.path.join(pre_DRR_path, suf, f), os.path.join(DRR_path,s, new_f))
                        shutil.copy(os.path.join(pre_DRR_path, suf, f), os.path.join(dataset_path,s, new_totall))
                        if suf in ['XY','YX']:
                            regularize_orientation([os.path.join(DRR_path,s, new_f),
                                                    os.path.join(dataset_path,s, new_totall)])
                        j += 1
                        in_dir_totall +=1
        totall += in_dir_totall
