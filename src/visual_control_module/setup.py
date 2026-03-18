import os
from glob import glob
from setuptools import setup

pkg = 'visual_control_module'

setup(
    name=pkg,
    version='0.0.0',
    packages=['vision_core'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + pkg]),
        ('share/' + pkg, ['package.xml']),
        (os.path.join('share', pkg, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='g1',
    maintainer_email='g1@todo.todo',
    description='JdeRobot GSoC 2026',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'visual_node = vision_core.visual_node:main',
        'data_recorder = vision_core.data_recorder:main',
        'visual_node_v2 = vision_core.visual_node_v2:main',
    ],
    },
)