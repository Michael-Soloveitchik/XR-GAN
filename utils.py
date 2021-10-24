import os
import shutil
# mkdirs = lambda x: os.path.exists(x) or os.makedirs()
# rmdirs = lambda x: (not os.path.exists(x)) or shutil.rmtree(x)
remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)