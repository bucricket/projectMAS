#!/usr/bin/env python

try:
    from setuptools import setup
    setup_kwargs = {'entry_points': {'console_scripts':['pydisalexi=pydisalexi.pydisalexi:main']}}
except ImportError:
    from distutils.core import setup
    setup_kwargs = {'scripts': ['bin/pydisalexi']}
    
from pydisalexi import __version__

setup(
    name="pydisalexi",
    version=__version__,
    description="An open source implementation of DisALEXI",
    author="Mitchell Schull",
    author_email="mitch.schull@noaa.gov",
    url="https://github.com/bucricket/pyDisALEXI",
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
