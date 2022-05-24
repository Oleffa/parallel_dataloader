import torch
import pytest
import psutil
import os
import shutil
import numpy as np
import h5py
from src.dataloader import DataLoader

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tmpdir = './tmp/'
features_dir = tmpdir + 'features/'

def get_dataset_features(dataset_size, sequence_size=2, feature_dims=[1], subset=0, \
        vram=100, ram=100, differently_sized=False):
    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    # Create some fake data
    fake_data_features = np.ones(tuple([dataset_size, sequence_size] + feature_dims), dtype=float)
    if differently_sized:
        fake_data_labels = np.ones((dataset_size-5, sequence_size, 1), dtype=np.uint8)
    else:
        fake_data_labels = np.ones((dataset_size, sequence_size, 1), dtype=np.uint8)
    with h5py.File(features_dir + 'data.h5', 'w') as f:
        f.create_dataset('features', data=fake_data_features)
        f.create_dataset('labels', data=fake_data_labels)

    dataset_params = {'ram' : ram,
        'vram' : vram,
        'data_path' : features_dir,
        'data_labels' : ['features', 'labels'],
        'data_types' : [float, np.uint8],
        'batch_size' : 32,
        'subset' : subset,
        }

    loader, dataset = DataLoader(dataset_params).get_loader()
    return loader, dataset, dataset_params

def test_1D():
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0)
    assert dataset.__len__() == dataset_size
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))

def test_1D_subset():
    subset = 10
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=subset)
    assert dataset.__len__() == subset
    amount_of_data = 0
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
        amount_of_data += data['features'].shape[0]
    assert amount_of_data == subset

def test_1D_subset_too_large():
    subset = 101
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    with pytest.raises(AssertionError, match='Error, subset larger than amount of data'):
        loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
                sequence_size=seq_size, feature_dims=feature_dims, subset=subset)
        for i, data in enumerate(loader, 0):
            pass

def test_1D_different_size():
    subset = 10
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    with pytest.raises(AssertionError, match='Error, some parts of the dataset have different amounts of datapoints'):
        loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
                sequence_size=seq_size, feature_dims=feature_dims, subset=subset, \
                differently_sized=True)
        for i, data in enumerate(loader, 0):
            pass

def test_2D():
    dataset_size = 100
    feature_dims = [10, 10]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))

def test_3D():
    dataset_size = 100
    feature_dims = [10, 10, 3]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))

def test_large_no_fit():
    dataset_size = 1000
    feature_dims = [100]
    seq_size = 200
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    assert dataset.fit_ram == False
    assert dataset.fit_vram == False

def test_large_ram_fit():
    dataset_size = 1000
    feature_dims = [100]
    seq_size = 200
    vram = 100.0
    ram = 1000.0
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0, \
            vram=vram, ram=ram)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    assert dataset.fit_ram == True
    assert dataset.fit_vram == False
    # TODO large files and VRAm/RAM loading
    # TODO test with only 1 feature and 3 features
    # Assert that the specified data_type in dataset_params is what comes from the hdf5
