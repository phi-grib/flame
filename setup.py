from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

import pathlib
import sys
import platform
import os
import shutil

class CustomInstall(install):
    def run(self):
        install.run(self)
        install.announce(self, 'Creating model repository folder...')
        if platform.system() == 'Windows':
          repo_path = pathlib.Path('C:/Users/Biel/Desktop/MODELS2')  #placeholder
        elif platform.system() == 'Darwin':  # mac os
          repo_path = pathlib.Path('~').expanduser() / 'flame_models'  #placeholder
        elif platform.system() == 'Linux':
          repo_path = pathlib.Path('~').expanduser() / 'flame_models' #placeholder
        repo_path.mkdir()

class CustomDevelopInstall(develop):
   def run(self):
        develop.run(self)
        develop.announce(self, 'Creating model repository folder...', 2)

        if platform.system() == 'Windows':
          repo_path = pathlib.Path('C:/Users/Biel/Desktop/MODELS4')  #placeholder
        elif platform.system() == 'Darwin':  # mac os
          repo_path = pathlib.Path('~').expanduser() / 'flame_models'  #placeholder
        elif platform.system() == 'Linux':
          repo_path = pathlib.Path('~').expanduser() / 'flame_models' #placeholder
        
        repo_path.mkdir()
        develop.announce(self, 'Copying conguration file...', 2)
        shutil.copy('./flame/config.yaml', repo_path)


models_path = '/home/biel/documents/'


setup(name='flame',
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
      package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.yaml', '*.yml'],
        },
      cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelopInstall
        }
      )
