import os
import psutil
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    r""" A custom dataset capable of respecting RAM and VRAM size

    Args:
        dataset_params: a dictionary of dataset parameters 
        train_params: a dictionary of parameters containing training
            parameters
    Shape:
    Attributes:
    Examples:
    """

    def __init__(self, dataset_params):
        self.__shuffle = dataset_params['shuffle']
        self.__data_path = dataset_params['data_path']
        self.__subset = dataset_params['subset']
        # Make list of all files that make up the dataset
        self.__file_load_list, self.__file_lists = self.get_file_list()

    def get_free_vram(self):
        #TODO

    def get_free_ram(self, return_swap=False):
        r""" Returns the amount of free RAM
        """
        swap = psutil.swap_memory()._asdict()['free'] / 1024.0 / 1024.0
        ram = psutil.virtual_memory()._asdict()['available'] / 1024.0 / 1024.0
        if return_swap:
            return ram + swap
        else:
            return ram

    def get_file_list(self):
        file_lists = []
        for dp in self.__data_path:
            file_lists.append([dp+ d for d in sorted(os.listdir(dp))])
        # Check that there are even amounts of files
        len_first = len(file_lists[0])
        assert all(len(i) == len_first for i in file_lists), "Error," \
                + " make sure there is an even amount of files for each data" \
                + " type in the dataset"
        file_load_list = np.arange(0, len(file_lists[-1]), 1)
        if self.__shuffle:
            np.random.shuffle(file_load_list)
        return file_load_list, file_lists

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        None

dataset_params_train = {'name' : 'train',                                  
    'ram' : 2000.0,
    'vram' : 200.0,
    'resize_features' : (100, 100),
    'shuffle' : True,
    'subset' : 1,
    'check_full_dataset' : False,
    'data_path' : ['/Volumes/Data/Google Drive/My Drive/Datasets/blob/blob_20/train/img/'],
    'data_labels' : ['images'],
    'data_types' : [np.uint8],
    'batch_size' : 32, 
    'verbose' : False,
}
custom_dataset = CustomDataset(dataset_params_train)
print(custom_dataset.get_free_ram())
print(custom_dataset.get_free_ram(return_swap=True))

