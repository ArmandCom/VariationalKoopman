import numpy as np
import os

path = '/data/BDSC/datasets/SSM_synced'


listdirs = [item for item in os.listdir(path)]
print(listdirs)
sets = []
for dir in listdirs:
    sets.append([os.path.join(path, dir, set) for set in os.listdir(os.path.join(path, dir))])

print(sets)
for set in sets:
    for subset in set:
        data = np.load(subset)
        print(data['marker_data'].shape, data['poses'].shape, data['marker_labels'].shape)
        # break