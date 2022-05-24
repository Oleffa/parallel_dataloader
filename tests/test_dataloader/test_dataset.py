import os
import shutil
import numpy as np
import cv2
from src.dataloader import DataSet

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
    return l


def get_test_blob_dataset():
    dataset_path = '/media/oli/LinuxData/datasets/'
    dataset_params = {'name' : 'train',
        'ram' : 4000.0,
        'vram' : 4000.0,
        'resize_function' : resize_on_load_blob,
        'shuffle' : True,
        'subset' : 1,
        'check_full_dataset' : True,
        'data_path' : [dataset_path + '/blob/blob_20/train/img/', \
                dataset_path + '/blob/blob_20/train/pos/', \
                dataset_path + '/blob/blob_20/train/vel/'],
        'data_labels' : ['features', 'pos', 'vel'],
        'data_types' : [np.uint8, float, float],
        'batch_size' : 32, #32 pairs of images = 64 batch size actually
        'verbose' : False,
        'device' : 'cuda',
        }
    return DataSet(dataset_params)

def test_blob():
    ds = get_test_blob_dataset()
    ds.stop()

def test_features():
    # Create tempdir
    tmpdir = './tmp/'
    features_dir = tmpdir + 'features/'
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)
    os.mkdir(features_dir)
    # Make up some data
    for i in range(3):
        np.save(features_dir + 'data{}.npy'.format(i), np.ones((100, 10, 50), dtype=np.uint8))

    dataset_params = {'name' : 'test',
        'ram' : 100.0,
        'vram' : 100.0,
        'resize_function' : resize_on_load_features,
        'shuffle' : False,
        'datapoints' : 100,
        'check_full_dataset' : True,
        'data_path' : [features_dir],
        'data_labels' : ['features'],
        'data_types' : [np.uint8],
        'batch_size' : 32,
        'verbose' : False,
        'device' : 'cuda',
        }
    ds = DataSet(dataset_params)
