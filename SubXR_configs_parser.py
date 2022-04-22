import pandas as pd
import json
import os
class SubXRParser():

    def __init__(self):
        with open(r"SubXR_configs_Masks", 'r') as f:
            self.configs = json.load(f)
    def __getitem__(self, key):
        if key in self.configs:
            val = self.configs[key]

            # if val suspected to be a path
            if key != 'mount':
                for prefix in ['google_drive_prefix_path','data_prefix_path', 'dataset_prefix_path', 'code_prefix_path']:
                    str_val = str(val)
                    if prefix in str_val:
                        val = eval(str_val.replace(prefix,self.configs["mount"][prefix]))
                self.configs[key] = val
            return self.configs[key]
        else:
            []