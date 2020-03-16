import sys, os
import torch
import numpy as np
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.omni_tools import copy2cpu as c2c


expr_code = 'poses'
msg = ''' Initial use of standard AMASS dataset preparation pipeline '''
amass_dir = '/data/BDSC/datasets/DFaust_67/*/*_poses.npz'
work_dir = '/data/BDSC/datasets/AMASS/'

logger = log2file(os.path.join(work_dir, '%s.log' % (expr_code)))
logger('[%s] AMASS Data Preparation Began.'%expr_code)
logger(msg)

#Note: should download all data of https://amass.is.tue.mpg.de/dataset
amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
              'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

from amass.prepare_data import prepare_amass
prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)