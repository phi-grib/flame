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
            repo_path = pathlib.Path('~/flame_models').expanduser()
        elif platform.system() == 'Darwin':  # mac os
            repo_path = pathlib.Path('~/flame_models').expanduser()
        elif platform.system() == 'Linux':
            repo_path = pathlib.Path('~/flame_models').expanduser()

        return repo_path

    def run(self):
        install.run(self)
        install.announce(self, 'Creating model repository folder...', 2)
        repo_path = self.get_repo_path()

        if repo_path.exists():
            install.warn(self, f'{repo_path} already exists')
        else:
            repo_path.mkdir()
            install.announce(self, f'Folder created at {str(repo_path)}', 2)
        # Modify conf.yaml with new default path using
        # the just installed flame manage

        from flame.manage import set_model_repository

        set_model_repository(repo_path)
        install.announce(self, f'Flame configuration updated succesfully', 2)


class CustomDevelopInstall(develop):
    """ Custom develop class to create the model repository directory
    during the develop installation mode
    """

    def run(self):
        develop.run(self)
        develop.announce(self, 'Creating model repository folder...', 2)
        # repo_path = self.get_repo_path()

        import appdirs
        repo_path = appdirs.user_data_dir('flame')
        repo_path = pathlib.Path(repo_path)
        if repo_path.exists():
            develop.warn(self, f'{repo_path} already exists')
        else:
            repo_path.mkdir(parents=True)
            develop.announce(self, f'Folder created at {str(repo_path)}', 2)

        from flame.manage import set_model_repository

        set_model_repository(repo_path)
        develop.announce(self, f'Flame configuration updated succesfully', 2)


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
    },
    install_requires=['appdirs', 'numpy', 'pandas']
)
