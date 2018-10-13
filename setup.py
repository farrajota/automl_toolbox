#!/usr/bin/env python


import os
from setuptools import setup, find_packages


# set version
VERSION = '0.0.2'


def get_requirements():
    """Loads contents from requirements.txt."""
    requirements = []
    with open('requirements.txt') as f:
        data = f.read().splitlines()
    if any(data):
        data = data[1:]
        requirements = [item.split(";")[0].split(" ")[0] for item in data]
    return requirements


LONG_DESCRIPTION = ''

readme_note = """\
.. note::
   For the latest source, discussion, etc, please visit the
   `GitHub repository <https://github.com/farrajota/automl_toolbox>`_\n\n
"""

with open('README.md') as fobj:
    LONG_DESCRIPTION = readme_note + fobj.read()


setup(
    name='automl_toolbox',
    version=VERSION,
    author='M. Farrajota',
    url='https://github.com/farrajota/automl_toolbox',
    download_url='',
    description="Toolbox for building automatic Data Science solutions",
    long_description=LONG_DESCRIPTION,
    license='MIT License',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    platforms='any',
    packages=find_packages(exclude=['docs',
                                    'notebooks',
                                    'tests']),
    install_requires=get_requirements(),
)
