import torch
import time
import pytest
import psutil
import os
import shutil
import numpy as np
import h5py
from parallel_dataloader.dataloader import DataLoader, FullLoader

tmpdir = './tmp/'
features_dir = tmpdir + 'features/'
torch.manual_seed(0)
np.random.seed(0)

def get_dataset_1features(dataset_size, files=1, sequence_size=2, feature_dims=[1], \
        subset=0, memory=100, differently_sized=False, device='cpu', shuffle=False, \
        dtype=float, reshape=None):
    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    # Create some fake data
    fake_data_features = np.zeros(tuple([dataset_size, sequence_size] + feature_dims), dtype=float)
    with h5py.File(features_dir + 'data.h5', 'w') as f:
        f.create_dataset('/data/features/0', data=fake_data_features)
    if files > 1:
        for i in range(1,files,1):
            with h5py.File(features_dir + 'data.h5', 'a') as f:
                f.create_dataset(f'/data/features/{i}', data=fake_data_features)
    del fake_data_features

    r = [feature_dims]
    if reshape is not None:
        r = reshape

    dataset_params = {'memory' : memory,
        'data_path' : features_dir,
        'data_labels' : ['features'],
        'data_types' : [dtype],
        'data_shapes' : [(tuple([sequence_size]+feature_dims))],
        'batch_size' : 32,
        'subset' : subset,
        'shuffle' : shuffle,
        'device' : device,
        'verbose' : False,
        'reshape' : r,
        }
    loader, dataset = DataLoader(dataset_params).get_loader()
    return loader, dataset, dataset_params

def get_dataset_numbered_features(dataset_size, files=1, sequence_size=2, feature_dims=[1], \
        subset=0, memory=100, differently_sized=False, device='cpu', shuffle=False, \
        dtype=float, reshape=None):
    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    # Create some fake data
    if files > 1:
        for i in range(0,files,1):
            a = np.arange(0, dataset_size*sequence_size*np.prod(feature_dims), 1, dtype=float) + \
                    dataset_size * sequence_size * np.prod(feature_dims) * i
            fake_data_features = a.reshape(tuple([dataset_size, sequence_size] + feature_dims))
            with h5py.File(features_dir + 'data.h5', 'a') as f:
                f.create_dataset(f'/data/features/{i}', data=fake_data_features)
            with h5py.File(features_dir + 'data.h5', 'a') as f:
                f.create_dataset(f'/data/labels/{i}', data=fake_data_features)

    del fake_data_features

    r = [feature_dims, feature_dims]
    if reshape is not None:
        r = reshape

    dataset_params = {'memory' : memory,
        'data_path' : features_dir,
        'data_labels' : ['features', 'labels'],
        'data_types' : [dtype, dtype],
        'data_shapes' : [tuple([sequence_size]+feature_dims), tuple([sequence_size]+feature_dims)],
        'batch_size' : 32,
        'subset' : subset,
        'shuffle' : shuffle,
        'device' : device,
        'verbose' : False,
        'reshape' : r,
        }

    loader, dataset = DataLoader(dataset_params).get_loader()
    return loader, dataset, dataset_params

def get_dataset_features(dataset_size, files=1, sequence_size=2, feature_dims=[1], \
        subset=0, memory=100, differently_sized=False, device='cpu', shuffle=False, \
        dtype=float, reshape=None, break_label=False):
    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    # Create some fake data
    fake_data_features = np.zeros(tuple([dataset_size, sequence_size] + feature_dims), dtype=float)
    if differently_sized:
        fake_data_labels = np.ones((dataset_size-5, sequence_size, 1), dtype=np.uint8)
    else:
        fake_data_labels = np.ones((dataset_size, sequence_size, 1), dtype=np.uint8)
    with h5py.File(features_dir + 'data.h5', 'w') as f:
        f.create_dataset('/data/features/0', data=fake_data_features)
        f.create_dataset('/data/labels/0', data=fake_data_labels)
    if files > 1:
        for i in range(1,files,1):
            with h5py.File(features_dir + 'data.h5', 'a') as f:
                f.create_dataset(f'/data/features/{i}', data=fake_data_features)
                f.create_dataset(f'/data/labels/{i}', data=fake_data_labels)
    del fake_data_labels
    del fake_data_features

    r = [feature_dims, [1]]
    if reshape is not None:
        r = reshape

    data_labels = ['features', 'labels']
    if break_label:
        data_labels = ['features' , 'img']

    dataset_params = {'memory' : memory,
        'data_path' : features_dir,
        'data_labels' : data_labels,
        'data_types' : [float, np.uint8],
        'data_shapes' : [(tuple([sequence_size]+feature_dims)), (sequence_size, 1)],
        'batch_size' : 32,
        'subset' : subset,
        'shuffle' : shuffle,
        'device' : device,
        'verbose' : False,
        'reshape' : r,
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
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_1D_reshape():
    dataset_size = 100
    feature_dims = [5]
    reshape = [[3], [1]]
    seq_size = 2
    with pytest.raises(AssertionError, match='Error, cant reshape 1D features!'):
        loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
                sequence_size=seq_size, feature_dims=feature_dims, subset=0, \
                reshape=reshape)
        for i, data in enumerate(loader, 0):
            assert (np.array(data['features'].shape) == [32, 2, 3]).any()
            break
        dataset.memory_loader.stop()
        time.sleep(0.5)
        assert dataset.memory_loader.is_alive() == False

def test_1D_broken_data_labels():
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    with pytest.raises(AssertionError, match='Error, passed data_labels that dont exist!'):
        loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
                sequence_size=seq_size, feature_dims=feature_dims, subset=0, break_label=True)
        for i, data in enumerate(loader, 0):
            pass
        dataset.memory_loader.stop()
        time.sleep(0.2)
        assert dataset.memory_loader.is_alive() == False

def test_1D_different_dtype():
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0, dtype=np.float16)
    for i, data in enumerate(loader, 0):
        pass
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_1D_multiple_files():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0)
    assert dataset.__len__() == files*dataset_size
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_1D_multiple_files_only1feature():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_1features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0)
    for i, data in enumerate(loader, 0):
        assert 'labels' not in data, "Error, found unexpected data label"
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_1D_multiple_files_multiple_epochs():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0)
    assert dataset.__len__() == files*dataset_size
    for e in range(5):
        amount_of_data = 0
        for i, data in enumerate(loader, 0):
            assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
                data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
            amount_of_data += data['features'].shape[0]
        assert amount_of_data == len(dataset)
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_1D_multiple_files_gpu():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, \
            device=device)
    assert dataset.__len__() == files*dataset_size
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

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
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False
    
def test_1D_subset_not_too_large():
    files = 3
    subset = 101
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    with pytest.raises(AssertionError, match='Error, subset larger than amount of data'):
        loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
                sequence_size=seq_size, feature_dims=feature_dims, subset=subset)
        for i, data in enumerate(loader, 0):
            pass
        dataset.memory_loader.stop()
        time.sleep(0.2)
        assert dataset.memory_loader.is_alive() == False

def test_1D_subset_too_large():
    files = 1
    subset = 101
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    with pytest.raises(AssertionError, match='Error, subset larger than amount of data'):
        loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
                sequence_size=seq_size, feature_dims=feature_dims, subset=subset, \
                files=files)
        for i, data in enumerate(loader, 0):
            pass
        dataset.memory_loader.stop()
        time.sleep(0.2)
        assert dataset.memory_loader.is_alive() == False

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
        dataset.memory_loader.stop()
        time.sleep(0.2)
        assert dataset.memory_loader.is_alive() == False

def test_2D():
    dataset_size = 100
    feature_dims = [10, 10]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_2D_reshape():
    dataset_size = 100
    feature_dims = [10, 10]
    reshape = [[5, 5], [1]]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0, \
            reshape=reshape)
    for i, data in enumerate(loader, 0):
        assert (np.array(data['features'].shape) == [32, 2, 5, 5]).any()
        break
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_3D():
    dataset_size = 100
    feature_dims = [10, 10, 3]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_3D_reshape():
    dataset_size = 100
    feature_dims = [10, 10, 3]
    reshape = [[5,5,3], [1]]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0, \
            reshape=reshape)
    for i, data in enumerate(loader, 0):
        assert (np.array(data['features'].shape) == [32, 2, 5, 5, 3]).any()
        break
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_single_no_fit():
    #In this test we test if one file is too large for the specified memory
    dataset_size = 500
    feature_dims = [100]
    seq_size = 200
    memory = 50.0
    with pytest.raises(AssertionError, match='Error, one file is bigger than Memory...'):
        loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
                sequence_size=seq_size, feature_dims=feature_dims, subset=0,\
                memory=memory)
        for i, data in enumerate(loader, 0):
            pass
        dataset.memory_loader.stop()
        time.sleep(0.2)
        assert dataset.memory_loader.is_alive() == False

def test_multiple_no_fit():
    # In this test we test if multiple small files that exceed memory crash
    files=2
    dataset_size = 500
    feature_dims = [100]
    seq_size = 200
    memory = 100.0
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0,\
            memory=memory, files=files)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    assert dataset.fit_memory == False
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_large_ram_fit():
    dataset_size = 100
    feature_dims = [100]
    seq_size = 200
    files=2
    memory = 200.0
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0, \
            memory=memory, files=files)
    for i, data in enumerate(loader, 0):
        assert data['features'].shape == (tuple([dp['batch_size'], seq_size] + feature_dims)) or \
            data['features'].shape == (tuple([int(dataset.__len__()%dp['batch_size']), seq_size] + feature_dims))
    assert dataset.fit_memory == True
    dataset.memory_loader.stop()
    time.sleep(0.2)
    assert dataset.memory_loader.is_alive() == False

def test_large_vram_fit():
    # Test if 4 large files that do not all fit in VRAM work
    dataset_size = 5000
    feature_dims = [100]
    seq_size = 200
    files=4
    memory = 2000.0
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0, \
            memory=memory, files=files, device='cpu')
    data_amount = 0
    for i, data in enumerate(loader, 0):
        data_amount += data['features'].shape[0]
    assert data_amount == files * dataset_size
    assert dataset.fit_memory == False
    dataset.memory_loader.stop()

    while dataset.memory_loader.is_alive():
        time.sleep(0.1)

    assert dataset.memory_loader.is_alive() == False

def test_large_vram_fit_shuffled():
    dataset_size = 5000
    feature_dims = [100]
    seq_size = 200
    files=4
    memory = 2000.0

    # True data
    unshuffled_features = np.zeros(tuple([dataset_size, seq_size] + \
            feature_dims), dtype=float)
    unshuffled_labels = np.ones((dataset_size, seq_size, 1), dtype=np.uint8)

    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, subset=0, \
            memory=memory, files=files, device='cpu', shuffle=True)
    data_amount = 0
    shuffled_features = np.zeros_like(unshuffled_features, dtype=float)
    for i, data in enumerate(loader, 0):
        pass

    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_1D_multiple_files_epoch_shuffling():
    # Test if multiple files are shuffled correctly
    # and different each epoch
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, shuffle=True)
    data = np.arange(0, files*dataset_size*seq_size, 1).reshape((200, 2, 1))
    for i, data in enumerate(loader, 0):
        if i == 0:
            assert data['features'][5,0] == 100
        if i == 1:
            assert data['features'][5,1] == 91

    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

# The followind tests are for the FullDataloader which is loading an entire hdf5 dataset and returns it

def test_full():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, shuffle=False)
    features = FullLoader(dp).get_data('features')
    assert features.shape == (dataset_size*files, seq_size, feature_dims[0])
def test_full_3D():
    files = 2
    dataset_size = 100
    feature_dims = [10, 10, 3]
    seq_size = 2
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, shuffle=False)
    features = FullLoader(dp).get_data('features')
    assert features.shape == tuple(list((dataset_size*files, seq_size)) + feature_dims)

def test_full_shuffle():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, shuffle=True)
    features = FullLoader(dp).get_data('features')
    assert (features[0] != [[0],[1]]).any()
    assert (features[10] == [[370],[371]]).any()

def test_full_3D_reshape():
    files = 2
    dataset_size = 100
    feature_dims = [10, 10, 3]
    reshape= [[5, 5, 3], [1]]
    seq_size = 2
    loader, dataset, dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, shuffle=False, reshape=reshape)
    features = FullLoader(dp).get_data('features')
    assert features.shape == tuple(list((dataset_size*files, seq_size)) + reshape[0])
