import numpy as np
import torch
import os
import math
import threading 
import time
from queue import Queue
import psutil

class RAMLoader(threading.Thread):
    def __init__(self, ds, ram, files, file_load_list, load_per_file):
        threading.Thread.__init__(self)
        self.ram = ram
        self.ram_size = 0.0
        self.verbose = ds['verbose']
        self.max_ram_size = ds['ram']
        self.next_file = 0
        self.files = files
        self.file_load_list = file_load_list
        self.compression = ds['compression']
        self.running = True
        self.data_types = ds['data_types']
        self.data_labels = ds['data_labels']
        self.resize_on_load = ds['resize_function']
        self.load_per_file = load_per_file
        # Pausing thread
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())

    def load_files(self, index):
        """
        Loads a file based on its compression
        """
        translated_idx = self.file_load_list[index]
        fnames = []
        data = []
        for i, f in enumerate(self.files):
            fnames.append(f[translated_idx])
            if self.compression == 'gzip':
                l = np.load(f[translated_idx])['a'].astype(self.data_types[i])
            elif self.compression == 'none':
                l = np.load(f[translated_idx]).astype(self.data_types[i])
            if self.data_labels[i] == 'images':
                # Take care of the fact that sometimes datasets are not sequenced
                l = resize_on_load(l, self.data_types[i])
            data.append(l[:load_per_file].copy())
        fsize = sum([j.nbytes/1000000.0 for j in data])
        return data, fnames, fsize

    def get_ram_size(self):
        """
        Computes the total amount of data in RAM in mb
        """
        ram_size = 0
        for i in list(self.ram.queue):
            ram_size += i['size']
        return ram_size

    def stop(self):
        """
        Public function to stop the RAMLoader and free its memory
        """
        self.running = False

    def pause(self):
        self.paused = True
        self.pause_cond.acquire()

    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()

    def run(self):
        i = 0
        if self.verbose:
            print("Starting RAMLoader!")
        while self.running:
            self.ram_size = self.get_ram_size()
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
            while self.ram_size + self.max_fsize < self.max_ram_size:
                if not self.running:
                    break
                if self.next_file >= len(self.files[0]):
                    self.next_file = 0
                data, files, fsize = self.load_files(self.next_file)
                f = {'data' : data,
                        'size' : fsize,
                        'files' : files
                        }
                del data
                # If there is still space put the file to the ram
                self.next_file += 1
                self.ram.put(f)
                self.ram_size = self.get_ram_size()
            time.sleep(0.1)

        if self.running == True:
            if self.verbose:
                print("Warning, RAMLoader ended unintentionally!")
        else:
            if self.verbose:
                print("RAMLoader ended as expected!")

class DataLoader():
    def __init__(self, dataset_params):
        self.ds = dataset_params
        # Reset a variable which will be recomputed but can cause problems down the line with subset sizes
        if 'data_points_per_file' in self.ds:
            del self.ds['data_points_per_file']
        self.dataset= DataSet(self.ds)
        # shuffle = False: We shuffle data our own way
        self.loader = torch.utils.data.DataLoader(self.dataset, \
                batch_size=self.ds['batch_size'], shuffle=False)

    def get_loader(self):
        return self.loader, self.dataset

class DataSet():
    def __init__(self, dataset_params):
        self.ds = dataset_params
        self.v = self.ds['verbose']
        self.resize_on_load = self.ds['resize_function']

        # Make list of all files
        self.file_load_list, self.file_lists = self.get_file_list()

        # Compute how many files fit in ram and vram 
        self.ram_max_files, self.vram_max_files, fsize = self.check_dataset()
        self.ram_queue_size = math.floor(self.ram_max_files / self.vram_max_files)

        # Handle subset loading to load partials of the dataset
        self.len = self.ds['data_points_per_file'] * len(self.file_lists[0])

        # Determine how many datapoints fit in vram, used for the load lists
        self.vram_len = np.minimum(len(self.file_lists[0]), self.vram_max_files) \
                * self.ds['data_points_per_file']
        if self.v:
            print('Detected {} files, each with size {}MB.'\
                    .format(len(self.file_lists[0]), fsize))

        load_per_file = self.ds['data_points_per_file']
        print(self.len)
        exit()

        # Multithreading
        self.ram = Queue()
        self.ram_loader = RAMLoader(self.ds, self.ram, self.file_lists, \
                self.file_load_list, load_per_file)
        self.ram_loader.start()

        self.vram = None
        self.vram_size = 0.0
        self.vram_batch = 0
        self.process = psutil.Process(os.getpid())

    def detect_compression(self, f):
        file_ending = f.split('.')[-1]
        if file_ending == 'npz':
            self.ds['compression'] = 'gzip'
        elif file_ending == 'npy':
            self.ds['compression'] = 'none'
        else:
            print("Error, unknown compression for file {}. I can only read npy and npz files!".format(f))
            exit()

    def check_dataset_helper(self, i, d):
        """
        This function goes through either one or all files of the dataset and checks if they have all the same
        amount of data points in them.
        Parameters:
        i: load index
        d: path to dataset file
        """

        # Detect the compression and load the numpy file accordingly
        self.detect_compression(d)
        if self.ds['compression'] == 'none':
            f = np.load(d)
        elif self.ds['compression'] == 'gzip':
            f = np.load(d)['a']

        if 'features' in self.ds['data_labels']:
            a = self.ds['data_labels'].index('features')
            if a == 0:
                #If we are taking care of the features, resize them
                f = self.resize_on_load(f, self.ds['data_types'][a])

        if 'data_points_per_file' in self.ds:
            assert self.ds['data_points_per_file'] == f.shape[0], \
                    "Error, file {} does have different number of datapoints" \
                    .format(d)
        else:
            self.ds['data_points_per_file'] = f.shape[0]

        # Sum up file size for this combination of dataset
        fsize = f.nbytes/1024.0/1024.0
        if 'a' in locals():
            del a
        del f
        """
        Files are first put into a RAM queue and then into VRAM, thus all files have to at least fit into RAM alone.
        Also the files have to fit into VRAM if that option is enabled.
        """ 
        assert fsize < self.ds['ram'], "Error, the dataset has files that do not even fit in the RAM in one piece. Consider splitting them up."
        assert fsize < self.ds['vram'], "Error, the dataset has files that do not fit in the VRAM in one piece. Consider splitting them up."
        if self.v:
            print("Files {}: {}/{} done!, Size: {}MB".format(d, i+1, \
                    len(self.file_lists[0]), fsize))
        return fsize

    def check_dataset(self):
        if self.v:
            print('Checking dataset integrity and loading dataset parameters from one or more dataset files ...')
        self.ds['compression'] = 'none'
        # First element is good enough, all files should have the same size 
        # after unpacking
        if self.ds['check_full_dataset']:
            # Check all elements
            for i, f in enumerate(zip(*self.file_lists)):
                fsize = self.check_dataset_helper(i, f[0])
        else:
            # Load first element and only check that
            f = list(zip(*self.file_lists))[0]
            fsize = self.check_dataset_helper(0, f[0])
        if self.v:
            for key, value in self.ds.items():
                print("{}: {}".format(key, value))
            print('\nDataset OK!\n')
        return math.floor(self.ds['ram']/fsize), math.floor(self.ds['vram']/fsize), \
                fsize

    def get_file_list(self):
        file_lists = []
        for data_path in self.ds['data_path']:
            file_lists.append([data_path + d for d in sorted(os.listdir(data_path))])
        # Check that there are even amounts of files
        len_first = len(file_lists[0])
        assert all(len(i) == len_first for i in file_lists), "Error, there make sure there is an even amount of files for each data type in the dataset"
        file_load_list = np.arange(0, len(file_lists[-1]), 1)
        if self.ds['shuffle']:
            np.random.shuffle(file_load_list)
        return file_load_list, file_lists

    def __len__(self):
        return self.len

    def stop(self):
        try:
            self.ram_loader.is_alive()
            self.ram_loader.stop()
            self.ram_loader.join()
            del self.ram_loader
        except:
            pass

    def check_if_fit_vram(self, cur_len):
        """
        This function checks if the data fit in the VRAM and then stops the RAMLoader
        """
        if cur_len >= len(self.file_lists[0]):
            if self.v:
                print("The whole dataset in {} of size {} fit into VRAM of size {}, Stopping RAMLoader and freeing {} of RAM" \
                        .format(self.ds['data_path'], self.vram_size, self.ds['vram'], self.ds['ram']))
            self.ram_loader.stop()
            return True
        return False

    def fill_vram(self):
        load_list = []
        while self.vram_size < self.ds['vram']:
            if self.ram.qsize() > 0:
                # Pausing ram loader so we dont blow the RAM limit
                self.ram_loader.pause()
                g = self.ram.get()
                if len(load_list) == 0:
                    assert g['size'] < self.ds['vram'], "Error, a batch of size {} does not fit in {} of VRAM".format(g['size'], self.ds['vram'])
                if self.vram_size + g['size'] > self.ds['vram']:
                    self.ram_loader.resume()
                    break
                for idx, i in enumerate(g['data']):
                    print(i.shape, self.ds['data_points_per_file'])
                    assert i.shape[0] == self.ds['data_points_per_file'], 'Error: Data in AM does not have the specified data points per load file! {}'.format(g['files'][0])
                    g['data'][idx] = torch.from_numpy(i.copy()).to(self.ds['device'])
                # Continue RAM loading after g is in VRAM
                self.ram_loader.resume()
                self.vram_size += g['size']
                load_list.append(g.copy())
                del g
                if self.check_if_fit_vram(len(load_list)) == True:
                    del self.ram_loader
                    break
            else:
                if self.v:
                    print("Waiting to fetch more data into VRAM, only {}/{} used." \
                            .format(self.vram_size, self.ds['vram']))
                time.sleep(0.1)
        self.vram = load_list.copy()
        assert len(self.vram) != self.vram_len, "Error, something went wrong when loading data to VRAM"
        del load_list
        self.vram_batch += 1

    def reset_vram_load_list(self):
        # Make (random)load list depending on the maximum files that fit in the vram (when dataset is larger than the VRAM)
        # or the length of the entire dataset (when it fits in VRAM)
        self.vram_load_list = np.arange(0, min(self.vram_max_files, len(self.file_lists[0]))*self.ds['data_points_per_file'], 1)
        if self.ds['shuffle']:
            np.random.shuffle(self.vram_load_list)

    def load_next_batch(self):
        # Free old memory
        del self.vram
        torch.cuda.empty_cache()
        self.vram_size = 0.0
        self.fill_vram()
        self.reset_vram_load_list()

    def __getitem__(self, idx):
        raw_idx = idx
        if self.vram is None:
            # Initialize VRAM and load list
            self.fill_vram()
            batch = 0
            self.vram_batch = 0
            self.reset_vram_load_list()
        else:
            batch = 0
            while idx >= self.vram_len:
                idx -= self.vram_len
                batch += 1
            # This needs to reset the vram batch after a wrap around in batches
            if batch == 0 and self.vram_batch > 0:
                self.vram_batch = -1
                self.load_next_batch()
        if batch > self.vram_batch:
            self.load_next_batch()
        # Specifically reset the vram load list. If there is only one file in VRAM
        # then the above does never trigger reset_vram_load_list
        if raw_idx == 0:
            self.reset_vram_load_list()
        # Translate idx to file to load
        translated_idx = self.vram_load_list[idx]
        sublist = 0
        while translated_idx >= self.ds['data_points_per_file']:
            sublist += 1
            translated_idx -= self.ds['data_points_per_file']
        sample = {}
        for i, d in enumerate(self.ds['data_labels']):
            sample[d] = self.vram[sublist]['data'][i][translated_idx]
        return sample
