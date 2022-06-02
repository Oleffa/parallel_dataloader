# parallel\_dataloader
This package containes a custom dataloader that loads datasets
in hdf5 format from a directory.
It is capable of loading large datasets to GPU or RAM by cycling chunks of
data through RAM or VRAM in a separate thread and as a result greatly
speeding up training.
For usage examples check out the tests.

## Installation
To install the dataloader use
```
pip install -e .
```
To install the dataloader with testing capabilities use
```
pip install -e .[test]
```
Note, you might have to write
```
pip install -e '.[test]'
```

## Testing
Run tests with:
```
python -m pytest tests
```
Or with printouts:
```
python -m pytest -s tests
```
