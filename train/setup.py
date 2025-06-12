import os
from setuptools import find_packages
from setuptools import setup
import setuptools


import subprocess


REQUIRED_PACKAGES = [
]

setup(
    name='sony_angels_torch_trainer_single_node_single_gpu_static',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Sony Angels Torch Trainer using a Single Node with a Single GPU preloading all the images.'
)