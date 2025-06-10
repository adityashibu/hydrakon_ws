from setuptools import setup

package_name = 'control_module'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/pid_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/control_system.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_team',
    maintainer_email='team@formula-student.com',
    description='Formula Student ADS-DV Control Module',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speed_processor = control_module.speed_processor_node:main',
            'pid_controller = control_module.pid_controller_node:main',
            'vehicle_interface = control_module.vehicle_interface_node:main',
        ],
    },
)