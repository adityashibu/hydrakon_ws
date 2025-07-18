from setuptools import setup
from glob import glob
import os

package_name = 'perception_module'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/perception_launch.py']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='Perception fusion using LiDAR and ZED',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'perception_node = perception_module.perception_node:main',
            'navsat_transform_node = perception_module.navsat_transform_node:main',
        ],
    },
)
