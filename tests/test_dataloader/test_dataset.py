import torch
import time
import psutil
import os
import shutil
import numpy as np
import cv2
from src.dataloader import DataLoader

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tmpdir = './tmp/'
features_dir = tmpdir + 'features/'

def resize_on_load_blob(l, t):
    shape = (100, 100)
    out_shape = tuple(list(l.shape)[:-len(list(shape))] + list(shape))
    temp = np.zeros(out_shape).astype(t)
    for idx_j, j in enumerate(l):
        for idx_k, k in enumerate(j):
            temp[idx_j, idx_k] = cv2.resize(src=k, dsize=(shape[1], \
            shape[0]), interpolation=cv2.INTER_LINEAR)
    return temp

def resize_on_load_features(l, t):
    return l.astype(t)

def get_dataset_features(files, size, test_datapoint, subset=1):
    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    fake_data = np.ones((size, 10, 50), dtype=float)
    print("Creating fake dataset of size {}MB".format(files*(fake_data.nbytes/1024.0/1024.0)))
    # Make up some data
    for f in range(files):
        fake_data = np.ones((size, 10, 50), dtype=np.uint8)
        fake_data [test_datapoint, :, :] = 2
        np.save(features_dir + 'data{}.npy'.format(f), fake_data)

    dataset_params = {'ram' : 100.0,
        'vram' : 100.0,
        'resize_function' : resize_on_load_features,
        'shuffle' : False,
        'subset' : subset,
        'check_full_dataset' : True,
        'data_path' : [features_dir],
        'data_labels' : ['features'],
        'data_types' : [float],
        'batch_size' : 32,
        'verbose' : False,
        'device' : device,
        }
    loader, dataset = DataLoader(dataset_params).get_loader()
    return loader, dataset, dataset_params

def test_stop():
    loader, dataset, dp = get_dataset_features(files=1, size=100, test_datapoint=10)
    dataset.stop()
    assert hasattr(dataset, 'ram_loader') == False

def test_features_1():
    test_datapoint = 10
    loader, dataset, dp = get_dataset_features(files=1, size=100, test_datapoint=test_datapoint)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (data['features'].shape[0], 10, 50)
        if i == 0:
            assert data['features'][test_datapoint][0,0] == 2
    dataset.stop()
    assert hasattr(dataset, 'ram_loader') == False

def test_features_1_subset():
    test_datapoint = 10
    subset = 20
    loader, dataset, dp = get_dataset_features(files=1, size=100, test_datapoint=test_datapoint, subset=subset)
    for i, data in enumerate(loader, 0):
        pass
    dataset.stop()
    assert hasattr(dataset, 'ram_loader') == False
"""
def test_features_1_large():
    test_datapoint = 10
    loader, dataset, dp = get_dataset_features(files=100, size=100, test_datapoint=test_datapoint)
    for i, data in enumerate(loader, 0):
        if i == 0:
            assert data['features'][test_datapoint][0,0] == 2
    dataset.stop()
    assert hasattr(dataset, 'ram_loader') == False

def test_features_1_ramsize():
    # TODO
    free_ram = psutil.virtual_memory().available/1024.0/1024.0
    test_datapoint = 10
    loader, dataset, dp = get_dataset_features(files=100, size=2000, test_datapoint=test_datapoint)
    for i, data in enumerate(loader, 0):
        if i == 0:
            assert data['features'][test_datapoint][0,0] == 2
        free_ram_after = psutil.virtual_memory().available/1024.0/1024.0
        break
    dataset.stop()
    free_ram_after2 = psutil.virtual_memory().available/1024.0/1024.0
    assert hasattr(dataset, 'ram_loader') == False

def test_features_1_shuffle():
    test_datapoint = 10
    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    files = 50
    size = 10
    fake_data = np.ones((size, 1, 1), dtype=np.uint8)
    print("Creating fake dataset of size {}MB".format(files*(fake_data.nbytes/1024.0/1024.0)))
    # Make up some data
    fake_data_all = []
    for f in range(files):
        fake_data = np.ones((size, 1, 1), dtype=np.uint8)
        fake_data *= f
        fake_data_all.append(fake_data)
        np.save(features_dir + 'data{}.npy'.format(f), fake_data)

    dataset_params_normal = {'ram' : 100.0,
        'vram' : 100.0,
        'resize_function' : resize_on_load_features,
        'shuffle' : False,
        'check_full_dataset' : True,
        'data_path' : [features_dir],
        'data_labels' : ['features'],
        'data_types' : [float],
        'batch_size' : 4,
        'verbose' : False,
        'device' : device,
        }
    loader_normal, dataset_normal = DataLoader(dataset_params_normal).get_loader()

    dataset_params_shuffle = {'ram' : 100.0,
        'vram' : 100.0,
        'resize_function' : resize_on_load_features,
        'shuffle' : True,
        'check_full_dataset' : True,
        'data_path' : [features_dir],
        'data_labels' : ['features'],
        'data_types' : [float],
        'batch_size' : 4,
        'verbose' : False,
        'device' : device,
        }

    loader_shuffle, dataset_shuffle = DataLoader(dataset_params_shuffle).get_loader()
    
    fake_data_all_normal = np.concatenate(fake_data_all, 0)
    fake_data_all = [fake_data_all[i] for i in dataset_shuffle.file_load_list]
    fake_data_all_bad_shuffle = np.concatenate(fake_data_all, 0)

    test_data = torch.zeros((files*size, 1, 1))
    for i, data in enumerate(loader_shuffle, 0):
        test_data[i*data['features'].shape[0]:(i+1)*data['features'].shape[0]] = data['features']
    assert not (test_data.numpy() == fake_data_all_normal).all()
    assert not (test_data.numpy() == fake_data_all_bad_shuffle).all()
    dataset_normal.stop()
    dataset_shuffle.stop()
    assert hasattr(dataset_normal, 'ram_loader') == False
    assert hasattr(dataset_shuffle, 'ram_loader') == False

def test_features_1_multiple_features():
    test_datapoint = 10
    tmpdir = './tmp/'
    features_dirs = [tmpdir + 'features/', tmpdir + 'features2/', tmpdir + 'features3/']

    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    for f in features_dirs:
        os.mkdir(f)
    files = 5
    size = 10
    fake_data = np.ones((size, 1, 1), dtype=float)
    print("Creating fake dataset of size {}MB".format(files*(fake_data.nbytes/1024.0/1024.0)))
    # Make up some data
    fake_data_all = []
    for f in range(files):
        fake_data = np.ones((size, 1, 1), dtype=np.uint8)
        fake_data *= f
        fake_data_all.append(fake_data)
        np.save(features_dirs[0] + 'data{}.npy'.format(f), fake_data)
        np.save(features_dirs[1] + 'data{}.npy'.format(f), fake_data)
        np.save(features_dirs[2] + 'data{}.npy'.format(f), fake_data)

    dataset_params = {'ram' : 100.0,
        'vram' : 100.0,
        'resize_function' : resize_on_load_features,
        'shuffle' : False,
        'check_full_dataset' : True,
        'data_path' : [features_dirs[0], features_dirs[1], features_dirs[2]],
        'data_labels' : ['features', 'test1', 'test2'],
        'data_types' : [float, np.float16, np.uint8],
        'batch_size' : 16,
        'verbose' : False,
        'device' : device,
        }
    loader, dataset = DataLoader(dataset_params).get_loader()
    for i, data in enumerate(loader, 0):
        for f in dataset_params['data_labels']:
            assert data[f].shape == (data[f].shape[0], 1, 1)

    dataset.stop()
    assert hasattr(dataset, 'ram_loader') == False

"""
shutil.rmtree(tmpdir, ignore_errors=True)

# TODO assert ram sizes 
# TODO assert vram sizes 
# TODO remove subset
# TODO test different sized files
