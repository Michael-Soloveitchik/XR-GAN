import os
import shutil
import numpy as np
# mkdirs = lambda x: os.path.exists(x) or os.makedirs()
# rmdirs = lambda x: (not os.path.exists(x)) or shutil.rmtree(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)
remove_and_create = lambda x: (shutil.rmtree(os.path.abspath(x), ignore_errors=True)) or create_if_not_exists(x)
dirs_content = lambda paths, random: zip(*[list(np.random.permutation(os.listdir(path))) if random else sorted(os.listdir(path)) for path in paths if os.path.exists(os.path.abspath(path))])
match_in_dir_2_out_dir = lambda x2y, random: zip(dirs_content(x2y[0],random),np.tile(x2y[1], (size_dir_content(x2y[0][0]),1)))
match_A_2_B_files = lambda A,B, random: zip(match_in_dir_2_out_dir(np.array(A).T,random),match_in_dir_2_out_dir(np.array(B).T,random))
# def match_A_2_B_files(A,B,random):
#     a = match_in_dir_2_out_dir(np.array(A).T, random)
#     b = match_in_dir_2_out_dir(np.array(B).T, random)
#     return zip(a,b)
#
# #
# def match_in_dir_2_out_dir(x2y, random):
#     res_1 = dirs_content(x2y[0],random)
#     res_2 = np.tile([x2y[1]], (size_dir_content(x2y[0][0]),1))
#     return zip(res_1,res_2)
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
