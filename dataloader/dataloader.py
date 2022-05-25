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

class DataLoader():
    def __init__(self, dataset_params):
        self.dp = dataset_params
        self.dataset= DataSet(self.dp)
        # shuffle = False: We shuffle data our own way
        self.loader = torch.utils.data.DataLoader(self.dataset, \
                batch_size=self.dp['batch_size'], shuffle=False)
    def get_loader(self):
        return self.loader, self.dataset

class MemoryLoader(threading.Thread):
    def __init__(self, memory_size, memory, one_file_size, data_info, fits_memory, \
            data_labels, shuffle, device, verbose):
        threading.Thread.__init__(self)
        self.data_labels = data_labels
        self.data_info = data_info
        self.memory_size = memory_size
        self.one_file_size = one_file_size
        self.memory = memory
        self.running = True
        self.fits_memory = fits_memory
        self.shuffle = shuffle
        self.device = device
        self.v = verbose

        self.load_list = np.arange(0, len(data_info[data_labels[0]]), 1)
        if self.shuffle:
            np.random.shuffle(self.load_list)
        self.load_idx = 0

        self.update_cur_memory_size()
        
    def update_cur_memory_size(self):
        """Computes the total amount of data in memory in mb and updates
        the local variable. Also checks for overflows.
        """
        self.cur_memory_size = 0.0
        for i in list(self.memory.queue):
            self.cur_memory_size += i['size']
        assert self.cur_memory_size <= self.memory_size

    def stop(self):
        self.running = False

    def load_file(self, dl, load_idx):
        data_name_target = self.data_info[dl][load_idx]['data_name']
        with h5py.File(self.data_info[dl][load_idx]['data_path']) as f:
            for gname, group in f.items():
                for data_label, group2 in group.items():
                    for data_name, ds in group2.items():
                        if data_label == dl and \
                                data_name == data_name_target:
                            d = np.array(ds)
                            if self.shuffle:
                                for i in d:
                                    np.random.shuffle(i)
                            d = torch.from_numpy(d).to(self.device)
                            return d, self.data_info[dl][load_idx]['fsize']

    def run(self):
        while self.running:
            # If one more file fits in the RAM
            self.update_cur_memory_size()
            while self.cur_memory_size + self.one_file_size <= self.memory_size:
                if not self.running:
                    break
                if self.load_idx >= len(self.load_list):
                    if self.fits_memory:
                        self.running = False
                        self.load_idx = 0
                        break
                    else:
                        self.load_idx = 0
                    self.update_cur_memory_size()
                # Load one more file
                d = dict()
                sample_size = 0.0 
                datapoints = 0
                for dl in self.data_labels:
                    d[dl], fsize = self.load_file(dl, self.load_list[self.load_idx])
                    datapoints = d[dl].shape[0]
                    sample_size += fsize
                d['size'] = sample_size
                d['datapoints'] = datapoints
                self.memory.put(d)
                self.load_idx += 1
                self.update_cur_memory_size()
            time.sleep(0.1)
        if self.v:
            if self.fits_memory:
                print("Memory Loader ended as expected, everything fit into memory!")
            else:
                print("Memory Loader ended as expected!")

class DataSet(data.Dataset):
    def __init__(self, dataset_params):
        super().__init__()
        self.dp = dataset_params
        self.v = self.dp['verbose']

        # Get basic info about the dataset from get_files
        self.data_info = dict()
        for dl in self.dp['data_labels']:
            self.data_info[dl] = []
        self.len = 0
        self.dataset_size = 0.0
        self.dataset_files = 0
        self.get_files()

        self.one_file_size = 0.0
        for dl in self.dp['data_labels']:
            self.one_file_size += self.data_info[dl][0]['fsize']

        # Check if the full dataset fits in RAM
        self.fit_memory = False
        self.memory = Queue()
        if self.dp['device'] == 'cpu':
            assert self.one_file_size <= self.dp['memory'], \
                    "Error, one file is bigger than Memory..."
            if self.dataset_size < self.dp['memory']:
                self.fit_memory = True
            self.memory_loader = MemoryLoader(self.dp['memory'], self.memory, \
                    self.one_file_size, self.data_info, self.fit_memory, \
                    self.dp['data_labels'], self.dp['shuffle'], self.dp['device'], \
                    self.v)
        elif self.dp['device'] == 'cuda':
            assert self.one_file_size <= self.dp['memory'], \
                    "Error, one file is bigger than Memory..."
            if self.dataset_size < self.dp['memory']:
                self.fit_memory = True
            self.memory_loader = MemoryLoader(self.dp['memory'], self.memory, \
                    self.one_file_size, self.data_info, self.fit_memory, \
                    self.dp['data_labels'], self.dp['shuffle'], self.dp['device'], \
                    self.v)
        else:
            print("Error, unknown device: ", self.dp['device'])
            exit()

        self.memory_loader.start()

        self.cache = None
        self.cache_idx = 0

    def get_files(self):
        p = Path(self.dp['data_path'])
        assert p.is_dir()
        # Recursively find all h5 files
        files = sorted(p.glob('**/*.h5'))
        assert len(files) >= 1, 'No hdf5 datasets found in {}!'.format(p)
        shapes = None
        for p in files:
            with h5py.File(p) as f:
                for gname, group in f.items():
                    for data_label, group2 in group.items():
                        for data_name, ds in group2.items():
                            # Make sure all datasets have the same size
                            if shapes == None:
                                shapes = np.array(ds).shape
                            else:
                                assert shapes[0] == np.array(ds).shape[0], "Error, some parts " \
                                        + "of the dataset have different amounts of datapoints"
                            # Only take a part of the data when subset is specified
                            assert self.dp['subset'] <= shapes[0] and self.dp['subset'] >= 0, \
                                    "Error, subset larger than amount of data"

                            # Compute the amount of samples in the dataset
                            if self.dp['subset'] == 0:
                                if data_label == self.dp['data_labels'][0]:
                                    self.len += shapes[0]
                                    self.dataset_files += 1
                            else:
                                self.len = self.dp['subset']

                            fsize = np.array(ds[:self.len]).nbytes/1024.0/1024.0
                            self.dataset_size += fsize
                            self.data_info[data_label].append({
                                'data_path':p, 
                                'data_label': data_label,
                                'data_name': data_name,
                                'dtype': np.array(ds).dtype,
                                'shape': np.array(ds)[:self.len].shape, 
                                'fsize': fsize,
                                })

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.cache == None:
            self.cache = self.memory.get()
        elif idx-self.cache_idx >= self.cache['datapoints']:
            # If all fit in memory the memory loader ends.
            # So we have to put the taken file back into the queue 
            # or we run out of data
            if self.fit_memory:
                self.memory.put(self.cache)
            self.cache = self.memory.get()
            self.cache_idx += self.cache['datapoints']
        idx -= self.cache_idx
        d = dict()
        for dl in self.dp['data_labels']:
            d[dl] = self.cache[dl][idx]
        return d

