from setuptools import setup

setup(name='parallel-dataloader',
        version='1.0',
        install_requires=['torch==1.11', \
                'h5py==3.7.0', \
                'psutil==5.9.1', \
                'numpy==1.22.4', \
                'pytest==7.1.2'],
        py_modules=['dataloader']
)
