import struct
import numpy as np
import matplotlib as plt
import scipy.misc
from tqdm import tqdm
from os import makedirs
from os.path import join
from os.path import exists
from operator import itemgetter
import torch.utils.data as data
import torch
import random
from sklearn.neighbors import NearestNeighbors
import os
import imageio

def load_data_norb(root, is_train):


  if is_train:
    data_filename = 'data_train.npy'
    dist_filename = 'dist_train.npy'
    neigh_filename = 'neigh_train.npy'

  else:
    data_filename = 'data_test.npy'
    dist_filename = 'dist_test.npy'
    neigh_filename = 'neigh_test.npy'

  data_path = join(root, data_filename)
  dist_path = join(root, dist_filename)
  neigh_path = join(root, neigh_filename)
  inp = torch.FloatTensor(np.load(data_path))
  dist = torch.FloatTensor(np.load(dist_path))
  neigh = torch.LongTensor(np.load(neigh_path))

  return inp, neigh, dist

def prepare_data_norb(root, is_train):

    save = True
    if is_train:
        dat_filename = 'dat_data_train.npy'
        cat_filename = 'cat_data_train.npy'
        info_filename = 'info_data_train.npy'

    else:
        dat_filename = 'dat_data_test.npy'
        cat_filename = 'cat_data_test.npy'
        info_filename = 'info_data_test.npy'

    dat_path = join(root, dat_filename)
    cat_path = join(root, cat_filename)
    info_path = join(root, info_filename)
    inp = list(np.load(dat_path))
    cat = list(np.load(cat_path))
    info = list(np.load(info_path))
    len(info)

    # values2 = set(map(lambda x:x[1], info))
    # org_list_1 = [[[info[i], cat[i], inp[i]] for i in range(len(info)) if info[i][1] == x] for x in values2]
    # values = set(map(lambda x:x[0], info))
    # # [print() for y in org_list_1 for z in y]
    # org_list_2 = [[[z[0], z[1], z[2]] for y in org_list_1 for z in y if (z[0][0] == x and z[0][1] == xx)] for x in values for xx in values2]


    valinst = set(map(lambda x:x[0], info))
    valelev = set(map(lambda x: x[1], info))
    valazim = set(map(lambda x: x[2], info))
    vallight = set(map(lambda x: x[3], info))
    valcat = set(map(lambda x:x, cat))

    '''organize by azimuth'''
    org_list_1 = [[[info[i], cat[i], inp[2*i:(2*(i+1))]] for i in range(len(info))
                   if (info[i][0] == x and info[i][1] == xx and info[i][3] == xxx and cat[i] == xxxx)]
                  for x in valinst for xx in valelev for xxx in vallight for xxxx in valcat]
    [y.sort(key=lambda x: x[0][2]) for y in org_list_1]
    org_arr_1 = np.asarray(org_list_1)
    info_1 = org_arr_1[:, :, 0]
    cat_1 = org_arr_1[:, :, 1]

    org_list_1 = org_arr_1[:, :, 2].tolist()
    for i in range(len(org_list_1)):
        for n in range(len(org_list_1[0])):
            org_list_1[i][n] = np.stack(org_list_1[i][n], axis=0)
        org_list_1[i] = np.stack(org_list_1[i], axis=0)
    dat_1 = np.stack(org_list_1, axis=0)
    bs, t, n_c, x, y = dat_1.shape
    dat_1 = np.reshape(np.transpose(dat_1, (0,2,1,3,4)),(bs*n_c, t, x, y))

    savetest = np.concatenate(dat_1[3],axis=0)
    imageio.imwrite('/data/BDSC/datasets/small_norb/test_image.png', savetest)

    '''organize by elevation'''
    org_list_2 = [[[info[i], cat[i], inp[2*i:(2*(i+1))]] for i in range(len(info))
                   if (info[i][0] == x and info[i][2] == xx and info[i][3] == xxx and cat[i] == xxxx)]
                  for x in valinst for xx in valazim for xxx in vallight for xxxx in valcat]
    [y.sort(key=lambda x: x[0][1]) for y in org_list_2]
    org_arr_2 = np.asarray(org_list_2)
    info_2 = org_arr_2[:, :, 0]
    cat_2 = org_arr_2[:, :, 1]
    org_list_2 = org_arr_2[:, :, 2].tolist()

    for i in range(len(org_list_2)):
        for n in range(len(org_list_2[0])):
            org_list_2[i][n] = np.stack(org_list_2[i][n], axis=0)
        org_list_2[i] = np.stack(org_list_2[i], axis=0)
    dat_2 = np.stack(org_list_2, axis=0)
    bs, t, n_c, x, y = dat_2.shape
    dat_2 = np.reshape(np.transpose(dat_2, (0,2,1,3,4)),(bs*n_c, t, x, y))


    '''organize by lightning'''
    # org_list_3 = [[[info[i], cat[i], inp[2*i:(2*(i+1))]] for i in range(len(info))
    #                if (info[i][0] == x and info[i][1] == xx and info[i][2] == xxx and cat[i] == xxxx)]
    #               for x in valinst for xx in valelev for xxx in valazim for xxxx in valcat]
    # [y.sort(key=lambda x: x[0][3]) for y in org_list_3]
    # org_arr_3 = np.asarray(org_list_3)
    # info_3 = org_arr_3[:, :, 0]
    # cat_3 = org_arr_3[:, :, 1]
    # org_list_3 = org_arr_3[:, :, 2].tolist()
    #
    # for i in range(len(org_list_3)):
    #     for n in range(len(org_list_3[0])):
    #         org_list_3[i][n] = np.stack(org_list_3[i][n], axis=0)
    #     org_list_3[i] = np.stack(org_list_3[i], axis=0)
    # dat_3 = np.stack(org_list_3, axis=0)
    # bs, t, n_c, x, y = dat_3.shape
    # dat_3 = np.reshape(np.transpose(dat_3, (0,2,1,3,4)),(bs*n_c, t, x, y))


    # info = np.concatenate((info_1, info_2, info_3), axis=0)
    # cat = np.concatenate((cat_1, cat_2, cat_3), axis = 0)
    # dat = np.concatenate((dat_1, dat_2, dat_3), axis = 0)

    if is_train:
        split = 'train'
    else:
        split = 'test'

    if save:
    # Note: Check if we can process different lengths dynamically --> we should!
        np.save('/data/BDSC/datasets/small_norb/preproc_data/dat_azim_proc_data_' + split, dat_1)
        np.save('/data/BDSC/datasets/small_norb/preproc_data/cat_azim_proc_data_' + split, cat_1)
        np.save('/data/BDSC/datasets/small_norb/preproc_data/info_azim_proc_data_' + split, info_1)

        np.save('/data/BDSC/datasets/small_norb/preproc_data/dat_elev_proc_data_' + split, dat_2)
        np.save('/data/BDSC/datasets/small_norb/preproc_data/cat_elev_proc_data_' + split, cat_2)
        np.save('/data/BDSC/datasets/small_norb/preproc_data/info_elev_proc_data_' + split, info_2)

        # np.save('/data/BDSC/datasets/small_norb/dat_light_proc_data_' + split, dat_3)
        # np.save('/data/BDSC/datasets/small_norb/cat_light_proc_data_' + split, cat_3)
        # np.save('/data/BDSC/datasets/small_norb/info_light_proc_data_' + split, info_3)
    else:
        print('Save is turned off')

def prepare_dataset_norb(root, is_train):

    save = True
    n_neighbors = 4
    traj_length = 9

    if is_train:
        dat_azim_filename = 'preproc_data/dat_azim_proc_data_train.npy'      # [1350, 18]
        dat_elev_filename = 'preproc_data/dat_elev_proc_data_train.npy'      # [2700,  9]
        # dat_elev_filename = 'dat_light_proc_data_train.npy'   # [4050,  6]

    else:
        dat_azim_filename = 'preproc_data/dat_azim_proc_data_test.npy'
        dat_elev_filename = 'preproc_data/dat_elev_proc_data_test.npy'
        # dat_elev_filename = 'dat_elev_light_data_train.npy'

    dat_azim_path = os.path.join(root, dat_azim_filename)
    dat_elev_path = os.path.join(root, dat_elev_filename)
    azim_data = np.load(dat_azim_path)/255.
    elev_data = np.load(dat_elev_path)/255.

    time_length = azim_data.shape[1]
    # assert traj_length%2 == 0

    # Note: take several sequences of T = trajectory length of the same time sequences for azimuth
    data = []
    stride = (traj_length // 2)
    # n_chunks = int(time_length // (stride))
    for i in range(0,time_length - traj_length, stride):
        new_inp = azim_data[:,i:i + traj_length]
        data.append(new_inp)
    data = np.concatenate(data, axis=0)

    # Note: Concatenate them with the elevation data
    data = np.concatenate([data, elev_data], axis=0)

    # Note: augment data:
    # By taking every two
    data_even = []
    data_odd = []
    for i in range(time_length):
        new_inp = azim_data[:,i:i+1]
        if i%2==0:
            data_even.append(new_inp)
        else:
            data_odd.append(new_inp)
    data_even = np.concatenate(data_even, axis=1)
    data_odd = np.concatenate(data_odd, axis=1)
    data_strided = np.concatenate([data_even, data_odd], axis=0)
    data = np.concatenate([data, data_strided], axis=0)

    # By flipping direction:
    data_reversed = np.flip(data, axis=1)
    data = np.concatenate([data, data_reversed], axis=0)

    # Note: Find nearest neighbors and distances
    train_distances_all = []
    train_indices_all = []

    for i in range(data.shape[0]):
        train_nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(data[i].reshape(traj_length, -1))
        train_distances, train_indices = train_nbrs.kneighbors(data[i].reshape(traj_length, -1))
        train_distances_all.append(train_distances)
        train_indices_all.append(train_indices)
    train_distances = np.stack(train_distances_all, axis=0)[:,:,1:]
    train_indices = np.stack(train_indices_all, axis=0)[:,:,1:]

    if is_train:
        split = 'train'
    else:
        split = 'test'
    if save:
        np.save(join(root,'data_')+split+'.npy', data)
        np.save(join(root,'neigh_')+split+'.npy', train_indices)
        np.save(join(root,'dist_')+split+'.npy', train_distances)
    else:
        print('Save is turned off')

def main():
    bools = [True, False]
    for bool in bools:
        prepare_data_norb('/data/BDSC/datasets/small_norb/preproc_data', bool) #TODO: also with False
        prepare_dataset_norb('/data/BDSC/datasets/small_norb/', bool)

    # prepare_dataset_norb('/data/BDSC/datasets/small_norb/', False)

    # X, neigh, dist = load_data_norb('/data/BDSC/datasets/small_norb/', True)
    # dset = data.TensorDataset(X, neigh, dist)
    #
    # for i, (x,n,d) in enumerate(data.DataLoader(dset, 2)):
    #   print(x, '\n', n, '\n', d, '\n')
    #   if i > 3:
    #     break

if __name__ == "__main__":
    main()



# class SmallNORBExample:
#
#     def __init__(self):
#         self.image_lt = None
#         self.image_rt = None
#         self.category = None
#         self.instance = None
#         self.elevation = None
#         self.azimuth = None
#         self.lighting = None
#
#     def __lt__(self, other):
#         return self.category < other.category or \
#                (self.category == other.category and self.instance < other.instance)
#
#     def show(self, subplots):
#         fig, axes = subplots
#         fig.suptitle(
#             'Category: {:02d} - Instance: {:02d} - Elevation: {:02d} - Azimuth: {:02d} - Lighting: {:02d}'.format(
#                 self.category, self.instance, self.elevation, self.azimuth, self.lighting))
#         axes[0].imshow(self.image_lt, cmap='gray')
#         axes[1].imshow(self.image_rt, cmap='gray')
#
#     @property
#     def pose(self):
#         return np.array([self.elevation, self.azimuth, self.lighting], dtype=np.float32)
#
#
# class SmallNORBDataset:
#     # Number of examples in both train and test set
#     n_examples = 24300
#
#     # Categories present in small NORB dataset
#     categories = ['animal', 'human', 'airplane', 'truck', 'car']
#
#     def __init__(self, dataset_root):
#         """
#         Initialize small NORB dataset wrapper
#
#         Parameters
#         ----------
#         dataset_root: str
#             Path to directory where small NORB archives have been extracted.
#         """
#
#         self.dataset_root = dataset_root
#         self.initialized = False
#
#         # Store path for each file in small NORB dataset (for compatibility the original filename is kept)
#         self.dataset_files = {
#             'train': {
#                 'cat': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'),
#                 'info': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'),
#                 'dat': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
#             },
#             'test': {
#                 'cat': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'),
#                 'info': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'),
#                 'dat': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
#             }
#         }
#
#         # Initialize both train and test data structures
#         self.data = {
#             'train': [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)],
#             'test': [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)]
#         }
#
#         # Fill data structures parsing dataset binary files
#         for data_split in ['train', 'test']:
#             self._fill_data_structures(data_split)
#
#         self.initialized = True
#
#     def explore_random_examples(self, dataset_split):
#         """
#         Visualize random examples for dataset exploration purposes
#
#         Parameters
#         ----------
#         dataset_split: str
#             Dataset split, can be either 'train' or 'test'
#         Returns
#         -------
#         None
#         """
#         if self.initialized:
#             subplots = plt.subplots(nrows=1, ncols=2)
#             for i in np.random.permutation(SmallNORBDataset.n_examples):
#                 self.data[dataset_split][i].show(subplots)
#                 plt.waitforbuttonpress()
#                 plt.cla()
#
#     def export_to_jpg(self, export_dir):
#         """
#         Export all dataset images to `export_dir` directory
#
#         Parameters
#         ----------
#         export_dir: str
#             Path to export directory (which is created if nonexistent)
#
#         Returns
#         -------
#         None
#         """
#         if self.initialized:
#             print('Exporting images to {}...'.format(export_dir), end='', flush=True)
#             for split_name in ['train', 'test']:
#
#                 split_dir = join(export_dir, split_name)
#                 if not exists(split_dir):
#                     makedirs(split_dir)
#
#                 for i, norb_example in enumerate(self.data[split_name]):
#                     category = SmallNORBDataset.categories[norb_example.category]
#                     instance = norb_example.instance
#
#                     image_lt_path = join(split_dir, '{:06d}_{}_{:02d}_lt.jpg'.format(i, category, instance))
#                     image_rt_path = join(split_dir, '{:06d}_{}_{:02d}_rt.jpg'.format(i, category, instance))
#
#                     scipy.misc.imsave(image_lt_path, norb_example.image_lt)
#                     scipy.misc.imsave(image_rt_path, norb_example.image_rt)
#             print('Done.')
#
#     def group_dataset_by_category_and_instance(self, dataset_split):
#         """
#         Group small NORB dataset for (category, instance) key
#
#         Parameters
#         ----------
#         dataset_split: str
#             Dataset split, can be either 'train' or 'test'
#         Returns
#         -------
#         groups: list
#             List of 25 groups of 972 elements each. All examples of each group are
#             from the same category and instance
#         """
#         if dataset_split not in ['train', 'test']:
#             raise ValueError('Dataset split "{}" not allowed.'.format(dataset_split))
#
#         groups = []
#         for key, group in groupby(iterable=sorted(self.data[dataset_split]),
#                                   key=lambda x: (x.category, x.instance)):
#             groups.append(list(group))
#
#         return groups
#
#     def _fill_data_structures(self, dataset_split):
#         """
#         Fill SmallNORBDataset data structures for a certain `dataset_split`.
#
#         This means all images, category and additional information are loaded from binary
#         files of the current split.
#
#         Parameters
#         ----------
#         dataset_split: str
#             Dataset split, can be either 'train' or 'test'
#         Returns
#         -------
#         None
#         """
#         dat_data = self._parse_NORB_dat_file(self.dataset_files[dataset_split]['dat'])
#         cat_data = self._parse_NORB_cat_file(self.dataset_files[dataset_split]['cat'])
#         info_data = self._parse_NORB_info_file(self.dataset_files[dataset_split]['info'])
#         np.save('/data/BDSC/datasets/small_norb/dat_data_'+dataset_split, dat_data)
#         np.save('/data/BDSC/datasets/small_norb/cat_data_'+dataset_split, cat_data)
#         np.save('/data/BDSC/datasets/small_norb/info_data_'+dataset_split, info_data)
#         for i, small_norb_example in enumerate(self.data[dataset_split]):
#             small_norb_example.image_lt = dat_data[2 * i]
#             small_norb_example.image_rt = dat_data[2 * i + 1]
#             small_norb_example.category = cat_data[i]
#             small_norb_example.instance = info_data[i][0]
#             small_norb_example.elevation = info_data[i][1]
#             small_norb_example.azimuth = info_data[i][2]
#             small_norb_example.lighting = info_data[i][3]
#
#     @staticmethod
#     def matrix_type_from_magic(magic_number):
#         """
#         Get matrix data type from magic number
#         See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.
#         Parameters
#         ----------
#         magic_number: tuple
#             First 4 bytes read from small NORB files
#         Returns
#         -------
#         element type of the matrix
#         """
#         convention = {'1E3D4C51': 'single precision matrix',
#                       '1E3D4C52': 'packed matrix',
#                       '1E3D4C53': 'double precision matrix',
#                       '1E3D4C54': 'integer matrix',
#                       '1E3D4C55': 'byte matrix',
#                       '1E3D4C56': 'short matrix'}
#         magic_str = bytearray(reversed(magic_number)).hex().upper()
#         return convention[magic_str]
#
#     @staticmethod
#     def _parse_small_NORB_header(file_pointer):
#         """
#         Parse header of small NORB binary file
#
#         Parameters
#         ----------
#         file_pointer: BufferedReader
#             File pointer just opened in a small NORB binary file
#         Returns
#         -------
#         file_header_data: dict
#             Dictionary containing header information
#         """
#         # Read magic number
#         magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)
#
#         # Read dimensions
#         dimensions = []
#         num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
#         for _ in range(num_dims):
#             dimensions.extend(struct.unpack('<i', file_pointer.read(4)))
#
#         file_header_data = {'magic_number': magic,
#                             'matrix_type': SmallNORBDataset.matrix_type_from_magic(magic),
#                             'dimensions': dimensions}
#         return file_header_data
#
#     @staticmethod
#     def _parse_NORB_cat_file(file_path):
#         """
#         Parse small NORB category file
#
#         Parameters
#         ----------
#         file_path: str
#             Path of the small NORB `*-cat.mat` file
#         Returns
#         -------
#         examples: ndarray
#             Ndarray of shape (24300,) containing the category of each example
#         """
#         with open(file_path, mode='rb') as f:
#             header = SmallNORBDataset._parse_small_NORB_header(f)
#
#             num_examples, = header['dimensions']
#
#             struct.unpack('<BBBB', f.read(4))  # ignore this integer
#             struct.unpack('<BBBB', f.read(4))  # ignore this integer
#
#             examples = np.zeros(shape=num_examples, dtype=np.int32)
#             for i in tqdm(range(num_examples), desc='Loading categories...'):
#                 category, = struct.unpack('<i', f.read(4))
#                 examples[i] = category
#
#             return examples
#
#     @staticmethod
#     def _parse_NORB_dat_file(file_path):
#         """
#         Parse small NORB data file
#         Parameters
#         ----------
#         file_path: str
#             Path of the small NORB `*-dat.mat` file
#         Returns
#         -------
#         examples: ndarray
#             Ndarray of shape (48600, 96, 96) containing images couples. Each image couple
#             is stored in position [i, :, :] and [i+1, :, :]
#         """
#         with open(file_path, mode='rb') as f:
#             header = SmallNORBDataset._parse_small_NORB_header(f)
#
#             num_examples, channels, height, width = header['dimensions']
#
#             examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)
#
#             for i in tqdm(range(num_examples * channels), desc='Loading images...'):
#                 # Read raw image data and restore shape as appropriate
#                 image = struct.unpack('<' + height * width * 'B', f.read(height * width))
#                 image = np.uint8(np.reshape(image, newshape=(height, width)))
#
#                 examples[i] = image
#
#         return examples
#
#     @staticmethod
#     def _parse_NORB_info_file(file_path):
#         """
#         Parse small NORB information file
#         Parameters
#         ----------
#         file_path: str
#             Path of the small NORB `*-info.mat` file
#         Returns
#         -------
#         examples: ndarray
#             Ndarray of shape (24300,4) containing the additional info of each example.
#
#              - column 1: the instance in the category (0 to 9)
#              - column 2: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70
#                degrees from the horizontal respectively)
#              - column 3: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
#              - column 4: the lighting condition (0 to 5)
#         """
#         with open(file_path, mode='rb') as f:
#
#             header = SmallNORBDataset._parse_small_NORB_header(f)
#
#             struct.unpack('<BBBB', f.read(4))  # ignore this integer
#
#             num_examples, num_info = header['dimensions']
#
#             examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)
#
#             for r in tqdm(range(num_examples), desc='Loading info...'):
#                 for c in range(num_info):
#                     info, = struct.unpack('<i', f.read(4))
#                     examples[r, c] = info
#
#         return examples