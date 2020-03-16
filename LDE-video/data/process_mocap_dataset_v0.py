import numpy as np
import os
import random
from sklearn.neighbors import NearestNeighbors

save = True

# path = '/data/BDSC/datasets/DFaust_Synthetic_Mocap'
path = '/data/BDSC/datasets/DFaust_Synthetic_Mocap/'
listdirs = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path,item))]

traj_length = 16
perc_train = 80
n_neighbors = 4

data = []
def split_data(inp, data, perc_train, traj_length):
    time_length = inp.shape[0]
    dim = inp.shape[1]
    assert traj_length%2 == 0

    input = []
    n_chunks = int(time_length // (traj_length / 2))
    for i in range(n_chunks):
        new_inp = inp[i:i + traj_length]
        data.append(new_inp)

    # print(inp[:(-remain)].shape, inp.shape, remain)
    # inp[:(-remain-1)].reshape(int(time_length//(traj_length/2)), int(traj_length/2), dim)

    return data


sets = []
for dir in listdirs:
    sets.append([os.path.join(path, dir, set) for set in os.listdir(os.path.join(path, dir))])

for set in sets:
    for subset in set:
        if subset.endswith('.npz'): #and not subset.endswith('shape.npz')
            ori_data = np.load(subset)
            poses = ori_data['markers'][:, 3:66]
            poses = poses.reshape((poses.shape[0], 63*3))
        data = split_data(poses, data, perc_train=perc_train, traj_length=traj_length)

n_val = int(len(data)*0.2)
random.shuffle(data)
train_data = np.stack(data[n_val:], axis=0)
val_data = np.stack(data[:n_val], axis=0)

train_distances_all = []
train_indices_all = []
val_distances_all = []
val_indices_all = []

for i in range(train_data.shape[0]):
    train_nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(train_data[i])
    train_distances, train_indices = train_nbrs.kneighbors(train_data[i])
    train_distances_all.append(train_distances)
    train_indices_all.append(train_indices)
train_distances = np.stack(train_distances_all, axis=0)[:,:,1:]
train_indices = np.stack(train_indices_all, axis=0)[:,:,1:]

for i in range(val_data.shape[0]):
    val_nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(val_data[i])
    val_distances, val_indices = val_nbrs.kneighbors(val_data[i])
    val_distances_all.append(val_distances)
    val_indices_all.append(val_indices)
val_distances = np.stack(val_distances_all, axis=0)[:,:,1:]
val_indices = np.stack(val_indices_all, axis=0)[:,:,1:]

if save:
    np.save(os.path.join(path,'train_pose_data.npy'), train_data)
    np.save(os.path.join(path,'val_pose_data.npy'), val_data)
    np.save(os.path.join(path,'train_pose_neigh.npy'), train_indices)
    np.save(os.path.join(path,'val_pose_neigh.npy'), val_indices)
    np.save(os.path.join(path,'train_pose_dist.npy'), train_distances)
    np.save(os.path.join(path,'val_pose_dist.npy'), val_distances)

