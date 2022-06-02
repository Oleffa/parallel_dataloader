from setuptools import setup

setup(name='parallel-dataloader',
        version='1.0',
        packages=['parallel_dataloader'],
        install_requires=['torch==1.11', \
                'h5py==3.7.0', \
                'psutil==5.9.1', \
                'opencv-python==4.5.5.64', \
                'numpy==1.22.4'],
        extras_require={
            'test' : ['pytest==7.1.2'],
            },
)
