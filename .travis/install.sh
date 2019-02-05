if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    #Useful for debugging any issues with conda
    conda info -a
    conda env create -f environment.yml
    source activate flame
    python setup.py install

else
    # Install some custom requirements on Linux  - sudo apt-get update
    # We do this conditionally because it saves us some downloading if the
    # version is the same.
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    #Useful for debugging any issues with conda
    conda info -a
    conda env create -f environment.yml
    source activate flame
    python setup.py install
fi