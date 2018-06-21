from setuptools import setup, find_packages

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
        }
      )
