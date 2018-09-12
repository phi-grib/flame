from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

import pathlib
import sys
import platform
import os
import shutil


class CustomInstall(install):
    """ Custom installation class to create the model repository directory
    during the standard installation
    """

    def get_repo_path(self) -> pathlib.Path:
        """ Returns the path of the model repo dir """
        if platform.system() == 'Windows':
            # placeholders
            repo_path = pathlib.Path('~').expanduser() / 'flame_models'
        elif platform.system() == 'Darwin':  # mac os
            repo_path = pathlib.Path('~').expanduser() / 'flame_models'
        elif platform.system() == 'Linux':
            repo_path = pathlib.Path('~').expanduser() / 'flame_models'

        return repo_path

    def run(self):
        install.run(self)
        install.announce(self, 'Creating model repository folder...')
        repo_path = self.get_repo_path()
        repo_path.mkdir()


class CustomDevelopInstall(develop):
    """ Custom develop class to create the model repository directory
    during the develop installation mode
    """

    def get_repo_path(self) -> pathlib.Path:
        """ Returns the path of the model repo dir """
        if platform.system() == 'Windows':
            # placeholders
            repo_path = pathlib.Path('~').expanduser() / 'flame_models'
        elif platform.system() == 'Darwin':  # mac os
            repo_path = pathlib.Path('~').expanduser() / 'flame_models'
        elif platform.system() == 'Linux':
            repo_path = pathlib.Path('~').expanduser() / 'flame_models'

        return repo_path

    def run(self):
        develop.run(self)
        develop.announce(self, 'Creating model repository folder...', 2)
        repo_path = self.get_repo_path()
        repo_path.mkdir()
        develop.announce(self, 'Copying conguration file...', 2)
        shutil.copy('./flame/config.yaml', repo_path)


setup(
    name='flame',
    version='0.1',
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
    package_data={'': ['*.yaml', '*.yml']},
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelopInstall
    }
)
