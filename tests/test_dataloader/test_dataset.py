import torch
import time
import pytest
import psutil
import os
import shutil
import numpy as np
import h5py

tmpdir = './tmp/'
features_dir = tmpdir + 'features/'
torch.manual_seed(0)
np.random.seed(0)
from parallel_dataloader.dataloader import DataLoader, FullLoader

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
        dtype=float, reshape=None, return_dataloader=False):
    # Create tempdir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    # Create some fake data
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

    if return_dataloader:
        loader, dataset = DataLoader(dataset_params).get_loader()
        return loader, dataset, dataset_params
    else:
        return dataset_params

def get_dataset_features(dataset_size, files=1, sequence_size=2, feature_dims=[1], \
        subset=0, memory=100, differently_sized=False, device='cpu', shuffle=False, \
        dtype=float, reshape=None, break_label=False, return_dataloader=True):
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

    if return_dataloader:
        loader, dataset = DataLoader(dataset_params).get_loader()
        return loader, dataset, dataset_params
    else:
        return dataset_params

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

# The followind tests are for the FullDataloader which is loading an entire hdf5 dataset and returns it
def test_full():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, \
            shuffle=False, return_dataloader=False)
    features = FullLoader(dp).get_data('features')
    assert features.shape == (dataset_size*files, seq_size, feature_dims[0])

def test_full_3D():
    files = 2
    dataset_size = 100
    feature_dims = [10, 10, 3]
    seq_size = 2
    dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, \
            shuffle=False, return_dataloader=False)
    features = FullLoader(dp).get_data('features')
    assert features.shape == tuple(list((dataset_size*files, seq_size)) + feature_dims)

def test_full_3D_reshape():
    files = 2
    dataset_size = 100
    feature_dims = [10, 10, 3]
    reshape= [[5, 5, 3], [1]]
    seq_size = 2
    dp = get_dataset_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=2, subset=0, shuffle=False, \
            reshape=reshape, return_dataloader=False)
    features = FullLoader(dp).get_data('features')
    assert features.shape == tuple(list((dataset_size*files, seq_size)) + reshape[0])

def test_full_no_shuffle():
    files = 1
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=False, \
            return_dataloader=False)
    features = FullLoader(dp).get_data('features')
    assert (features[0] == [[0.],[1.]]).any()
    assert (features[10] == [[20.],[21.]]).any()

np.random.seed(0)
def test_full_shuffle():
    files = 1
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=True, \
            return_dataloader=False)
    features = FullLoader(dp).get_data('features')
    assert (features[0] == [[52.],[53.]]).any()
    assert (features[10] == [[106.],[107.]]).any()
    assert (features[20] == [[86.],[87.]]).any()

np.random.seed(0)
def test_full_shuffle_2_files():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=True)
    features = FullLoader(dp).get_data('features')
    assert (features[0] == [[132.],[133.]]).any()
    assert (features[10] == [[12.],[13.]]).any()
    assert (features[20] == [[204.],[205.]]).any()

np.random.seed(0)
def test_full_no_shuffle_2_files():
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=False)
    features = FullLoader(dp).get_data('features')
    assert (features[0] == [[0.],[1.]]).any()
    assert (features[101] == [[202.],[203.]]).any()

np.random.seed(0)
def test_1D_multiple_files_epoch_shuffling():
    # Test if multiple files are shuffled correctly
    # and different each epoch
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=True, \
            return_dataloader=True)
    
    for e in range(2):
        for i, data in enumerate(loader, 0):
            if i == 0 and e == 0:
                assert data['features'][5,0] == 380
            if i == 0 and e == 1:
                assert data['features'][5,0] == 388

    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

np.random.seed(0)
def test_1D_multiple_files_epoch_no_shuffling():
    # Test if multiple files are shuffled correctly
    # and different each epoch
    files = 2
    dataset_size = 100
    feature_dims = [1]
    seq_size = 2
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=False, \
            return_dataloader=True)

    epoch_data = []
    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['features'].numpy())
        loader_data = np.concatenate(loader_data, 0)
        epoch_data.append(loader_data)

    assert (epoch_data[0] == epoch_data[1]).all() == True
    assert (epoch_data[0] == epoch_data[2]).all() == True
    assert (epoch_data[1] == epoch_data[2]).all() == True

    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_if_all_there():
    files = 2
    dataset_size = 5
    feature_dims = [1]
    seq_size = 1
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=False, \
            return_dataloader=True)
    data = []
    for i in range(0,files,1):
        a = np.arange(0, dataset_size*seq_size*np.prod(feature_dims), 1, dtype=float) + \
                dataset_size * seq_size * np.prod(feature_dims) * i
        data.append(a.reshape(tuple([dataset_size, seq_size] + feature_dims)))
    data = np.concatenate(data, 0)

    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['features'].numpy())
        loader_data = np.concatenate(loader_data, 0)
        assert np.array_equal(data, loader_data) == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_if_all_there_subset():
    files = 3
    dataset_size = 10
    feature_dims = [1]
    seq_size = 1
    subset = 12
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=subset, shuffle=False, \
            return_dataloader=True)
    data = FullLoader(dp).get_data('features')

    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['features'].numpy())
        loader_data = np.concatenate(loader_data, 0)
        assert np.array_equal(data, loader_data) == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_if_all_there_shuffle():
    files = 2
    dataset_size = 5
    feature_dims = [1]
    seq_size = 1
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=True, \
            return_dataloader=True)

    epoch_data = []
    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['features'].numpy())
        loader_data = np.concatenate(loader_data, 0)
        epoch_data.append(loader_data)

    assert (epoch_data[0] == epoch_data[1]).all() == False
    assert (epoch_data[0] == epoch_data[2]).all() == False
    assert (epoch_data[1] == epoch_data[2]).all() == False
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_if_all_there_no_fit():
    files = 2
    dataset_size = 500
    feature_dims = [100]
    seq_size = 100
    memory = 100
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=False, \
            return_dataloader=True)
    data = []
    for i in range(0,files,1):
        a = np.arange(0, dataset_size*seq_size*np.prod(feature_dims), 1, dtype=float) + \
                dataset_size * seq_size * np.prod(feature_dims) * i
        data.append(a.reshape(tuple([dataset_size, seq_size] + feature_dims)))
    data = np.concatenate(data, 0)

    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['features'].numpy())
        loader_data = np.concatenate(loader_data, 0)
        assert np.array_equal(data, loader_data) == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_if_all_there_shuffle_no_fit():
    files = 2
    dataset_size = 500
    feature_dims = [100]
    seq_size = 100
    memory = 100
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=0, shuffle=True, \
            return_dataloader=True)

    epoch_data = []
    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['features'].numpy())
        loader_data = np.concatenate(loader_data, 0)
        epoch_data.append(loader_data)

    assert (epoch_data[0] == epoch_data[1]).all() == False
    assert (epoch_data[0] == epoch_data[2]).all() == False
    assert (epoch_data[1] == epoch_data[2]).all() == False
    assert np.array_equal(np.sort(epoch_data[0], 0), np.sort(epoch_data[1], 0)) == True
    assert np.array_equal(np.sort(epoch_data[0], 0), np.sort(epoch_data[2], 0)) == True
    assert np.array_equal(np.sort(epoch_data[2], 0), np.sort(epoch_data[1], 0)) == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_if_all_there_shuffle_subset_no_fit():
    files = 2
    dataset_size = 500
    feature_dims = [100]
    seq_size = 100
    memory = 100
    subset=750
    loader, dataset, dp = get_dataset_numbered_features(dataset_size=dataset_size, \
            sequence_size=seq_size, feature_dims=feature_dims, files=files, subset=750, shuffle=True, \
            return_dataloader=True)

    epoch_data = []
    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['features'].numpy())
        loader_data = np.concatenate(loader_data, 0)
        epoch_data.append(loader_data)

    assert (epoch_data[0] == epoch_data[1]).all() == False
    assert (epoch_data[0] == epoch_data[2]).all() == False
    assert (epoch_data[1] == epoch_data[2]).all() == False
    assert np.array_equal(np.sort(epoch_data[0], 0), np.sort(epoch_data[1], 0)) == True
    assert np.array_equal(np.sort(epoch_data[0], 0), np.sort(epoch_data[2], 0)) == True
    assert np.array_equal(np.sort(epoch_data[2], 0), np.sort(epoch_data[1], 0)) == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_cartpole():
    print("cartpole")
    dp = {'memory' : 100,
        'data_path' : '/media/oli/LinuxData/googledrive/datasets/data/LFO/cartpole/detenv/detpol/',
        'data_labels' : ['states', 'actions'],
        'data_types' : [float, int],
        'data_shapes' : [[2, 4], [2, 1]],
        'batch_size' : 32,
        'subset' : 0,
        'shuffle' : False,
        'device' : 'cpu',
        'verbose' : True,
        'reshape' : [[2, 4], [2, 1]],
        }

    states = FullLoader(dp).get_data('states')
    loader, dataset = DataLoader(dp).get_loader()

    data = []
    for i, d in enumerate(loader, 0):
        data.append(d['states'].cpu().numpy())
    data = np.concatenate(data, 0)
    assert (data == states).all() == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_blob():
    print("blob")
    dp = {'memory' : 2000,
        'data_path' : '/media/oli/LinuxData/googledrive/datasets/data/blob/blob_20/train/',
        'data_labels' : ['img', 'vel', 'pos'],
        'data_types' : [np.uint8, float, float],
        'data_shapes' : [[20, 100, 100], [20, 2], [20, 2]],
        'batch_size' : 64,
        'subset' : 0,
        'shuffle' : False,
        'device' : 'cpu',
        'verbose' : True,
        'reshape' : [[50, 50], [2], [2]],
        }

    states = FullLoader(dp).get_data('img')
    loader, dataset = DataLoader(dp).get_loader()

    data = []
    for i, d in enumerate(loader, 0):
        data.append(d['img'].cpu().numpy())
    data = np.concatenate(data, 0)
    assert (data == states).all() == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_blob_shuffle():
    print("blob shuffle")
    dp = {'memory' : 2000,
        'data_path' : '/media/oli/LinuxData/googledrive/datasets/data/blob/blob_20/train/',
        'data_labels' : ['img', 'vel', 'pos'],
        'data_types' : [np.uint8, float, float],
        'data_shapes' : [[20, 100, 100], [20, 2], [20, 2]],
        'batch_size' : 64,
        'subset' : 0,
        'shuffle' : True,
        'device' : 'cpu',
        'verbose' : True,
        'reshape' : [[50, 50], [2], [2]],
        }

    np.random.seed(0)
    states = FullLoader(dp).get_data('img')
    np.random.seed(0)
    loader, dataset = DataLoader(dp).get_loader()

    data = []
    for i, d in enumerate(loader, 0):
        data.append(d['img'].cpu().numpy())
    data = np.concatenate(data, 0)
    assert np.array_equal(np.sort(data, 0), np.sort(states, 0)) == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False


def test_pong():
    print("pong")
    dp = {'memory' : 7000,
        'data_path' : '/media/oli/LinuxData/googledrive/datasets/data/LFO/pong_visual/detenv/detpol/',
        'data_labels' : ['states', 'actions'],
        'data_types' : [np.uint8, int],
        'data_shapes' : [[2, 200, 200, 3], [2, 1]],
        'batch_size' : 64,
        'subset' : 0,
        'shuffle' : False,
        'device' : 'cpu',
        'verbose' : True,
        'reshape' : [[50, 50, 3], [1]],
        }

    states = FullLoader(dp).get_data('states')
    loader, dataset = DataLoader(dp).get_loader()

    data = []
    for i, d in enumerate(loader, 0):
        data.append(d['states'].cpu().numpy())
    data = np.concatenate(data, 0)
    assert (data == states).all() == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_pong_no_fit():
    print("pong")
    dp = {'memory' : 175,
        'data_path' : '/media/oli/LinuxData/googledrive/datasets/data/LFO/pong_visual/detenv/detpol/',
        'data_labels' : ['states', 'actions'],
        'data_types' : [np.uint8, int],
        'data_shapes' : [[2, 200, 200, 3], [2, 1]],
        'batch_size' : 64,
        'subset' : 0,
        'shuffle' : False,
        'device' : 'cpu',
        'verbose' : True,
        'reshape' : [[50, 50, 3], [1]],
        }

    states = FullLoader(dp).get_data('states')
    loader, dataset = DataLoader(dp).get_loader()

    import cv2
    for e in range(3):
        data = []
        for i, d in enumerate(loader, 0):
            data.append(d['states'].cpu().numpy())
        data = np.concatenate(data, 0)
        assert (data == states).all() == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_pong_no_fit_subset():
    print("pong")
    dp = {'memory' : 70,
        'data_path' : '/media/oli/LinuxData/googledrive/datasets/data/LFO/pong_visual/detenv/detpol/',
        'data_labels' : ['states', 'actions'],
        'data_types' : [np.uint8, int],
        'data_shapes' : [[2, 200, 200, 3], [2, 1]],
        'batch_size' : 64,
        'subset' : 5000,
        'shuffle' : False,
        'device' : 'cpu',
        'verbose' : True,
        'reshape' : [[50, 50, 3], [1]],
        }

    states = FullLoader(dp).get_data('states')
    loader, dataset = DataLoader(dp).get_loader()

    import cv2
    for e in range(3):
        data = []
        for i, d in enumerate(loader, 0):
            data.append(d['states'].cpu().numpy())
        data = np.concatenate(data, 0)
        assert (data == states).all() == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

def test_pong_no_fit_shuffle_subset():
    print("pong")
    dp = {'memory' : 70,
        'data_path' : '/media/oli/LinuxData/googledrive/datasets/data/LFO/pong_visual/detenv/detpol/',
        'data_labels' : ['states', 'actions'],
        'data_types' : [np.uint8, int],
        'data_shapes' : [[2, 200, 200, 3], [2, 1]],
        'batch_size' : 64,
        'subset' : 5000,
        'shuffle' : True,
        'device' : 'cpu',
        'verbose' : True,
        'reshape' : [[50, 50, 3], [1]],
        }

    states = FullLoader(dp).get_data('states')
    loader, dataset = DataLoader(dp).get_loader()

    import cv2
    epoch_data = []
    for e in range(3):
        loader_data = []
        for i, d in enumerate(loader, 0):
            loader_data.append(d['states'].numpy())
            #for x in d['states']:
            #    for s in x:
            #        print(s.shape)
            #        cv2.imshow("arst", s.cpu().numpy())
            #        cv2.waitKey(33)
        loader_data = np.concatenate(loader_data, 0)
        epoch_data.append(loader_data)

        # make sure that a random datapoint does not appear multiple times per epoch
        datapoint = loader_data[123]
        count = np.sum(np.all(loader_data == datapoint, axis=(1,2,3,4)))
        assert count == 1

    assert (epoch_data[0] == epoch_data[1]).all() == False
    assert (epoch_data[0] == epoch_data[2]).all() == False
    assert (epoch_data[1] == epoch_data[2]).all() == False
    assert np.array_equal(np.sort(epoch_data[0], 0), np.sort(epoch_data[1], 0)) == True
    assert np.array_equal(np.sort(epoch_data[0], 0), np.sort(epoch_data[2], 0)) == True
    assert np.array_equal(np.sort(epoch_data[2], 0), np.sort(epoch_data[1], 0)) == True
    dataset.memory_loader.stop()
    while dataset.memory_loader.is_alive():
        time.sleep(0.1)
    assert dataset.memory_loader.is_alive() == False

# Run all ILPO and VAE pre-train experiments with embeddings again
# Run VAE experiments again

# DONE was the data always the same???
#   - There was a problem when multiple files were loaded both when the data fit and didnt fit in memory
#   - Also the shuffle list was not always shuffled
#   - It is very likely that the VAEToolbox results and LFO results for pong_visual are skewed becaus of that
