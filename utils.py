import os
import shutil
import numpy as np
# mkdirs = lambda x: os.path.exists(x) or os.makedirs()
# rmdirs = lambda x: (not os.path.exists(x)) or shutil.rmtree(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)
remove_and_create = lambda x: (shutil.rmtree(os.path.abspath(x), ignore_errors=True)) or create_if_not_exists(x)
dirs_content = lambda paths, random: zip(*[list(np.random.permutation(os.listdir(path))) if random else sorted(os.listdir(path)) for path in paths if os.path.exists(os.path.abspath(path))])
# def dirs_content(paths, random):
#     res = []
#     for path in paths:
#         if os.path.exists(os.path.abspath(path)):
#             if random:
#                 res.append(np.random.permutation(os.listdir(path)))
#             else:
#                 res.append(sorted(os.listdir(path)))
#     return zip(*res)
size_dir_content = lambda path: len(os.listdir(path)) if os.path.exists(os.path.abspath(path)) else 0
