import numpy as np
import h5py
import torch
from torch.utils import data
from pathlib import Path
import os
import math
import threading 
import time
from queue import Queue
import psutil

class RAMLoader(threading.Thread):
    def __init__(self, ds, ram, files, file_load_list, load_per_file):
        pass
    def load_files(self, index):
        pass
    def get_ram_size(self):
        pass
    def stop(self):
        pass
    def pause(self):
        pass
    def resume(self):
        pass
    def run(self):
        pass

class DataLoader():
    def __init__(self, dataset_params):
        self.dp = dataset_params
        self.dataset= DataSet(self.dp)
        # shuffle = False: We shuffle data our own way
        self.loader = torch.utils.data.DataLoader(self.dataset, \
                batch_size=self.dp['batch_size'], shuffle=False)
    def get_loader(self):
        return self.loader, self.dataset

class DataSet(data.Dataset):
    def __init__(self, dataset_params):
        super().__init__()
        self.ds = dataset_params
        self.data_info = []
        self.get_files()
        self.get_total_data_size()
        self.fill_cache()

        # Check if the full dataset fits in RAM
        self.fit_ram = False
        self.fit_vram = False
        if self.get_total_data_size() < self.ds['ram']:
            self.fit_ram = True
        if self.get_total_data_size() < self.ds['vram']:
            self.fit_vram = True

    def get_total_data_size(self):
        return self.dataset_size

    def load_data(self, path, key, dtype):
        with h5py.File(path) as f:
            for dname, ds in f.items():
                if dname == key:
                    data = np.array(ds,dtype=dtype)
                    self.cache[key] = data[:self.len]

    def fill_cache(self):
        self.cache = dict()
        self.load_data(self.data_info[0]['data_path'], 'features', float)
        self.load_data(self.data_info[0]['data_path'], 'labels', np.uint8)

    def get_files(self):
        p = Path(self.ds['data_path'])
        assert p.is_dir()
        # Recursively find all h5 files
        files = sorted(p.glob('**/*.h5'))
        assert len(files) >= 1, 'No hdf5 datasets found in {}!'.format(p)
        dataset_size = 0
        shapes = None
        for p in files:
            with h5py.File(p) as f:
                for dname, ds in f.items():
                    idx = -1
                    # Make sure all datasets have the same size
                    if shapes == None:
                        shapes = np.array(ds).shape
                    else:
                        assert shapes[0] == np.array(ds).shape[0], "Error, some parts " \
                                + "of the dataset have different amounts of datapoints"
                    # Only take a part of the data when subset is specified
                    assert self.ds['subset'] <= shapes[0] and self.ds['subset'] >= 0, \
                            "Error, subset larger than amount of data"
                    if self.ds['subset'] == 0:
                        self.len = shapes[0]
                    else:
                        self.len = self.ds['subset']

                    fsize = np.array(ds[:self.len]).nbytes/1024.0/1024.0
                    dataset_size += fsize
                    self.data_info.append({'data_path': p, 'data_label': dname, 'dtype': np.array(ds).dtype, \
                            'shape': np.array(ds)[:self.len].shape, 'fsize': fsize})
        self.dataset_size = dataset_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = dict()
        for key in self.ds['data_labels']:
            sample[key] = self.cache[key][idx]
        return sample
