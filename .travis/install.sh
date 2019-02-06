if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --set always_yes yes
    conda update -q conda
    conda init
    source $HOME/.bash_profile
    #Useful for debugging any issues with conda
    conda info -a
    conda env create -f environment.yml
    conda activate flame
    echo "------FLAME ENV ACTIVATED--------"
    python setup.py install

else
    # Install some custom requirements on Linux  - sudo apt-get update
    # We do this conditionally because it saves us some downloading if the
    # version is the same.
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --set always_yes yes
    conda update -q conda
    conda init
    source $HOME/.bashrc
    #Useful for debugging any issues with conda
    conda info -a
    conda env create -f environment.yml
    conda activate flame
    echo "------FLAME ENV ACTIVATED--------"
    python setup.py install
fi