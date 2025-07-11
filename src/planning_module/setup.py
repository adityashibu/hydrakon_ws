from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'planning_module'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        ('share/' + package_name + '/launch', ['launch/planning_launch.py']),
        # Configuration files
        ('share/' + package_name + '/config', ['config/planning_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aditya S',
    maintainer_email='as2397@hw.ac.uk',
    description='Publishes target speed and steering for control module',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planning_node = planning_module.planning_node:main',
            # Keep legacy version for testing
            'planning_node_legacy = planning_module.planning_node:main',
        ],
    },
)