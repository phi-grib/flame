from setuptools import setup, find_packages
# from setuptools.command.install import install
# from setuptools.command.develop import develop

# import pathlib
# import sys
# import platform
# import os
# import shutil

setup(
    name='flame',
    version='1.0.0-rc4',
    license='GNU GPLv3 or posterior',
    description='',
    url='https://github.com/phi-grib/flame',
    download_url='https://github.com/phi-grib/flame.git',
    author='Manuel Pastor, Biel Stela, Jose Carlos Gomez',
    author_email='manuel.pastor@upf.edu',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['flame=flame.flame_scr:main'],
    },
    # If any package contains *.txt or *.rst files, include them:
    # package_data={'': ['*.yaml', '*.yml']},
    package_data={'flame': ['config.yaml','children/*.yaml', 'children/*.docx']},
    install_requires=['appdirs']
)
