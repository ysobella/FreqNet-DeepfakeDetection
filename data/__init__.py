import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import ConcatDataset
from .datasets import dataset_folder

'''
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
'''

import os
# def get_dataset(opt):
#     # classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
#     # if '0_real' not in classes or '1_fake' not in classes:
#     #     dset_lst = []
#     #     for cls in classes:
#     #         root = opt.dataroot + '/' + cls
#     #         dset = dataset_folder(opt, root)
#     #         dset_lst.append(dset)
#     #     return torch.utils.data.ConcatDataset(dset_lst)
#     # return dataset_folder(opt, opt.dataroot)
#     """
#         Loads dataset with 'real' and 'fake' subdirectories.
#         """
#     classes = ['real', 'fake']
#     dset_lst = []
#
#     for cls in classes:
#         root = os.path.join(opt.dataroot, cls)  # Ensure correct path
#         if os.path.exists(root):  # Ensure the directory exists
#             dset = dataset_folder(opt, root)
#             dset_lst.append(dset)
#
#     if not dset_lst:
#         raise ValueError(f"No valid dataset found in {opt.dataroot}")
#
#     return ConcatDataset(dset_lst)

def get_dataset(opt):
    """
    Loads dataset from the correct split (train, val, or test).
    """
    dataset_root = opt.dataroot  # Use exactly what is passed

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

    real_path = os.path.join(dataset_root, 'real')
    fake_path = os.path.join(dataset_root, 'fake')

    print(f"Final dataset root: {dataset_root}")
    print(f"init:Checking if '{real_path}' exists: {os.path.exists(real_path)}")
    print(f"init:Checking if '{fake_path}' exists: {os.path.exists(fake_path)}")

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(f"Expected 'real/' and 'fake/' inside {dataset_root}, but they are missing!")

    return dataset_folder(opt, dataset_root)  # Pass only the dataset root, not subfolders

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
