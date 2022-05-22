import numpy as np
from src.dataloader import CustomDataset

def test_get_file_list():
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
    file_load_list, file_lists = custom_dataset.get_file_list()
    print(file_load_list)
    assert 1 == 1
