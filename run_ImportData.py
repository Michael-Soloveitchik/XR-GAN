from utils import * #remove_and_create, shutil
from SubXR_configs_parser import SubXRParser
from tqdm import tqdm

def import_data(configs):
    for data_type in configs['Data']:
        for data_source in configs['Data'][data_type]:
            if configs['Data'][data_type][data_source]['in_dir']:
                if data_source == 'XR':
                    i = 0
                    for dir in tqdm(os.listdir(configs['Data'][data_type][data_source]['in_dir'])):
                        for im_path in os.listdir(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir)):
                            new_im_path = "xr_{idx}.jpg".format(idx=i)
                            shutil.copy(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir,im_path), os.path.join(configs['Data'][data_type][data_source]['out_dir'],new_im_path))
                            i += 1

if __name__ == '__main__':
    configs = SubXRParser()
    import_data(configs)
