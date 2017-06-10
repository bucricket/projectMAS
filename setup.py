#!/usr/bin/env python

import shutil
import os
try:
    from setuptools import setup
    setup_kwargs = {'entry_points': {'console_scripts':['pydisalexi=pydisalexi.pydisalexi_usda:main']}}
except ImportError:
    from distutils.core import setup
    setup_kwargs = {'scripts': ['bin/pydisalexi']}
    
from pydisalexi import __version__
prefix  = os.environ.get('PREFIX')
processDi = os.path.abspath(os.path.join(prefix,os.pardir))
processDir = os.path.join(processDi,'work')


disalexiPath = os.path.join(prefix,'share','disalexi')
if not os.path.exists(disalexiPath):
    os.makedirs(disalexiPath)
shutil.copyfile(os.path.join(processDir,'share','data','landcover.xlsx'),os.path.join(disalexiPath,'landcover.xlsx'))
setup(
    name="projectmas",
    version=__version__,
    description="An open source implementation of DisALEXI",
    author="Mitchell Schull",
    author_email="mitch.schull@noaa.gov",
    url="https://github.com/bucricket/projectMAS.git",
    packages= ['pydisalexi'],
    platforms='Posix; MacOS X; Windows',
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        # Uses dictionary comprehensions ==> 2.7 only
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: GIS',
    ],  
    **setup_kwargs
)
