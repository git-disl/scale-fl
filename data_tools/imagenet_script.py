import os
import pandas as pd
import sys
import glob
from scipy.io import loadmat

base_dir = '/Users/fatihilhan/Desktop/datasets/ImageNet'
os.chdir(base_dir)
data_path = 'train'
val_label_path = 'ILSVRC2012_validation_ground_truth.txt'

synsets = loadmat('meta.mat')['synsets']
with open('synset.txt') as f:
    synsets_map = f.readlines()
synsets_map = [v.split(' ')[0] for v in synsets_map]
synsets = [(synsets[i][0][0][0][0], synsets[i][0][1][0]) for i in range(1860)]

map_list = []
for synset in synsets:
    try:
        map_list.append(synsets_map.index(synset[1]))
    except:
        pass

val_labels = pd.read_csv(val_label_path, header=None)
val_labels = [v[0] for v in val_labels.values]

file_list = sorted(glob.glob(f"val/*.JPEG"))
for i, file_path in enumerate(file_list):
    n_str = synsets_map[map_list[val_labels[i]-1]]
    if not os.path.exists(f'val/{n_str}'):
        os.mkdir(f'val/{n_str}')
    try:
        file_name = os.path.split(file_path)[-1]
        os.rename(file_path, f'val/{n_str}/{file_name}')
    except:
        pass

# for i, folder_path in enumerate(sorted(os.listdir('train'))):
#     for file_name in sorted(os.listdir(f'train/{folder_path}')):
#         new_folder_path = file_name[:9]
#         if not os.path.isdir(f'train/{new_folder_path}'):
#             os.mkdir(f'train/{new_folder_path}')
#
#         os.rename(os.path.join(base_dir, 'train', folder_path, file_name), os.path.join(base_dir, 'train', new_folder_path, file_name))


# for i, folder_path in enumerate(sorted(os.listdir('val'))):
#     for file_name in sorted(os.listdir(f'val/{folder_path}')):
#         new_folder_path = synsets_map[map_list[val_labels[int(file_name[-10:-5])-1]-1]]
#         if not os.path.isdir(f'val/{new_folder_path}'):
#             os.mkdir(f'val/{new_folder_path}')
#
#         os.rename(os.path.join(base_dir, 'val', folder_path, file_name), os.path.join(base_dir, 'val', new_folder_path, file_name))