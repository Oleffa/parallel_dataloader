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

class FullLoader():
    def __init__(self, dataset_params):
        self.dp = dataset_params
        self.device = self.dp['device']
        self.data_info = dict()
        for dl in self.dp['data_labels']:
            self.data_info[dl] = []

    def load_file(self, dl, load_idx):
        data_name_target = self.data_info[dl][load_idx]['data_name']
        r = self.data_info[dl][load_idx]['reshape']
        with h5py.File(self.data_info[dl][load_idx]['data_path']) as f:
            for gname, group in f.items():
                for data_label, group2 in group.items():
                    for data_name, ds in group2.items():
                        if data_label == dl and \
                                data_name == data_name_target:
                            d = np.array(ds)
                            a = d.astype(self.data_info[dl][load_idx]['data_type'])
                            assert a.dtype == self.data_info[dl][load_idx]['data_type'], \
                                    "Error, loaded file was not converted to dtype specified" \
                                    + "in dataset_params['data_type']"
                            d = d.astype(self.data_info[dl][load_idx]['data_type'])
                            if tuple(r) != d.shape[-len(r):]:
                                d = reshape(d, r)
                            d = torch.from_numpy(d).to(self.device)
                            return d, self.data_info[dl][load_idx]['fsize']

    def get_data(self, dl):
        p = Path(self.dp['data_path'])
        assert p.is_dir(), "Error, {} is not a directory".format(p)
        # Recursively find all h5 files
        files = sorted(p.glob('**/*.h5'))
        assert len(files) >= 1, 'No hdf5 datasets found in {}!'.format(p)
        shapes = None
        for p in files:
            with h5py.File(p) as f:
                for gname, group in f.items():
                    for data_label, group2 in group.items():
                        for data_name, ds in group2.items():
                            if str(data_label) not in self.dp['data_labels']:
                                continue
                            # Make sure all datasets have the same size
                            if shapes == None:
                                shapes = np.array(ds).shape
                            else:
                                assert shapes[0] == np.array(ds).shape[0], "Error, some parts " \
                                        + "of the dataset have different amounts of datapoints"
                            data_type = self.dp['data_types'][self.dp['data_labels'].index(data_label)]
                            r = self.dp['reshape'][self.dp['data_labels'].index(data_label)]
                            fsize = np.zeros(list(ds.shape)[:2] + r, dtype=data_type).nbytes/1024.0/1024.0
                            r = self.dp['reshape'][self.dp['data_labels'].index(data_label)]
                            if r != list(np.array(ds).shape)[2:]:
                                assert len(r) == 2 or len(r) == 3 or len(r) == 4, "Error, cant reshape {} features!".\
                                        format(len(r))
                            self.data_info[data_label].append({
                                'data_path': p, 
                                'data_label': data_label,
                                'data_name': data_name,
                                'data_type': data_type,
                                'orig_shape': np.array(ds).shape, 
                                'reshape': r,
                                'fsize': fsize,
                                })
        info = self.data_info[dl]
        data = []
        for idx, i in enumerate(info):
            a = self.load_file(dl, idx)
            data.append(a[0])
        data = np.concatenate(data, axis=0)
        if self.dp['shuffle']:
            np.random.shuffle(data)
        return data

class DataLoader():
    def __init__(self, dataset_params):
        self.dp = dataset_params
        self.dataset = DataSet(self.dp)
        # shuffle = False: We shuffle data our own way
        self.loader = torch.utils.data.DataLoader(self.dataset, \
                batch_size=self.dp['batch_size'], shuffle=False)
    def get_loader(self):
        return self.loader, self.dataset

def reshape(d, r):
    """Resize a 2D or 3D feature with opencv
    """
    import cv2
    out_shape = tuple(list(d.shape)[:-len(r)] + r)
    tmp = np.zeros(out_shape).astype(d.dtype)
    for idx_j, j in enumerate(d):
        for idx_k, k in enumerate(j):
            tmp[idx_j, idx_k] = cv2.resize(src=k, dsize=tuple((r[1], r[0])), interpolation=cv2.INTER_LINEAR)
    return tmp


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

        # This load list is used to shuffle the loaded files
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

    def load_file(self, dl, load_idx, shuffle_list):
        data_name_target = self.data_info[dl][load_idx]['data_name']
        r = self.data_info[dl][load_idx]['reshape']
        with h5py.File(self.data_info[dl][load_idx]['data_path']) as f:
            for gname, group in f.items():
                for data_label, group2 in group.items():
                    for data_name, ds in group2.items():
                        if data_label == dl and \
                                data_name == data_name_target:
                            d = np.array(ds)
                            # Shuffle the data of an individual data file
                            if self.shuffle:
                                d = d[shuffle_list]
                            a = d.astype(self.data_info[dl][load_idx]['data_type'])
                            assert a.dtype == self.data_info[dl][load_idx]['data_type'], \
                                    "Error, loaded file was not converted to dtype specified" \
                                    + "in dataset_params['data_type']"
                            d = d.astype(self.data_info[dl][load_idx]['data_type'])
                            if tuple(r) != d.shape[-len(r):]:
                                d = reshape(d, r)
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
                # Get the number of elements in one data file
                a = self.data_info[self.data_labels[0]][0]['orig_shape'][0]
                # Create this shuffle list so all datafiles for a given data label
                # are shuffled in the same way
                shuffle_list = np.random.permutation(a)

                for dl in self.data_labels:
                    d[dl], fsize = self.load_file(dl, self.load_list[self.load_idx], shuffle_list)
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

        assert len(self.dp['data_labels']) == len(self.dp['data_types']), \
                "Error, length of data_labels != data_types"
        assert len(self.dp['data_labels']) == len(self.dp['data_shapes']), \
                "Error, length of data_labels != data_shapes"
        assert len(self.dp['data_types']) == len(self.dp['data_shapes']), \
                "Error, length of data_types != data_shapes"
        assert len(self.dp['data_labels']) == len(self.dp['reshape']), \
                "Error, length of data_labels != reshape"

        # Get basic info about the dataset from get_files
        self.data_info = dict()
        self.found_data_labels = []
        for dl in self.dp['data_labels']:
            self.data_info[dl] = []
        self.len = 0
        self.dataset_size = 0.0
        self.dataset_files = 0
        self.get_files()

        self.one_file_size = 0.0
        for dl in self.dp['data_labels']:
            assert str(dl) in self.found_data_labels, 'Error, passed data_labels that dont exist!'
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
        assert p.is_dir(), "Error, {} is not a directory".format(p)
        # Recursively find all h5 files
        files = sorted(p.glob('**/*.h5'))
        assert len(files) >= 1, 'No hdf5 datasets found in {}!'.format(p)
        shapes = None
        for p in files:
            with h5py.File(p) as f:
                for gname, group in f.items():
                    for data_label, group2 in group.items():
                        for data_name, ds in group2.items():
                            if str(data_label) not in self.found_data_labels:
                                self.found_data_labels.append(data_label)
                            if str(data_label) not in self.dp['data_labels']:
                                continue
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

                            data_type = self.dp['data_types'][self.dp['data_labels'].index(data_label)]
                            r = self.dp['reshape'][self.dp['data_labels'].index(data_label)]
                            fsize = np.zeros(list(ds.shape)[:2] + r, dtype=data_type).nbytes/1024.0/1024.0
                            self.dataset_size += fsize
                            r = self.dp['reshape'][self.dp['data_labels'].index(data_label)]
                            if r != list(np.array(ds).shape)[2:]:
                                assert len(r) == 2 or len(r) == 3, "Error, cant reshape 1D features!"
                            self.data_info[data_label].append({
                                'data_path': p, 
                                'data_label': data_label,
                                'data_name': data_name,
                                'data_type': data_type,
                                'orig_shape': np.array(ds)[:self.len].shape, 
                                'reshape': r,
                                'fsize': fsize,
                                })

            """
            assert self.dp['subset'] <= self.len, "Error, subset larger than dataset!"
            if self.dp['subset'] > 0:
                self.len = self.dp['subset']
            """

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx == 0:
            self.cache_idx = 0
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

