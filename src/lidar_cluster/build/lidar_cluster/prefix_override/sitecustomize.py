import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aditya/hydrakon_ws/src/lidar_cluster/install/lidar_cluster'
