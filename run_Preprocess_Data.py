import sys
import os
from Preprocessing.merge_script import merge_folders

if __name__ == '__main__':
    DATA_apth  = ' '.join(sys.argv[1:])
    merge_folders(DATA_apth)
