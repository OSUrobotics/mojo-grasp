from setuptools import setup

setup(
   name='mojograsp',
   version='0.1.0.devl',
   author='OSU GRASPING LAB',
   author_email='(Placeholder) navek@oregonstate.edu',
   packages=['mojograsp', 'mojograsp.simcore', 'mojograsp.simcore.simobject', 'mojograsp.simcore.sensors', 'mojograsp.simcore.actuators', 'mojograsp.simcore.datacollection', 'mojograsp.simcore.simmanager', ],
   license='LICENSE.txt',
   description='Mojo-Grasp library for grasping simulation written on top of pybullet',
   install_requires=[
   ],
)
