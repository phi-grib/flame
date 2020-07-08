#! -*- coding: utf-8 -*-

# Description    Context wrapps calls to predict and build to
# support models making use of extenal input sources
#
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
#
# Copyright 2018 Manuel Pastor
#
# This file is part of Flame
#
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
#
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
import appdirs
import os
import sys

from flame.util import utils

# def change_config_status() -> None:
#     """Changes config status in config.yml to True"""
#     config = utils._read_configuration()

#     if config['config_status']:
#         return
#     else:
#         config['config_status'] = True
#     utils.write_config(config)


def ask_user ():
    ''' utility function to obtain binary confirmation (yes/no) from the user
    '''
    userinput = input()

    if userinput.lower() not in ['yes', 'no', 'y', 'n']:
        print('Please write "yes", "no", "y" or "n"')
        return False
    elif userinput.lower() in ['yes', 'y']:
        return True
    return False


def configure(path: None, silent: False):
    """Configures model repository.

    Loads config.yaml and writes a correct model repository path
    with the path provided by the user or a default from appdirs
    if the path is not provided.
    """
    
    success, config = utils.read_config()
    if not success:
        return False, config

    if silent:
        if path is not None:  
            source_dir = os.path.abspath(path)
        else:
            source_dir = os.path.dirname(os.path.dirname(__file__)) 

        models_path = os.path.join (source_dir,'models')
        predictions_path = os.path.join (source_dir,'predictions')
        spaces_path = os.path.join (source_dir,'spaces')

        try:
            if not os.path.isdir(models_path):
                os.mkdir(models_path)
            if not os.path.isdir(spaces_path):
                os.mkdir(spaces_path)
            if not os.path.isdir(predictions_path):
                os.mkdir(predictions_path)
        except Exception as e:
            return False, e
        
        utils.set_repositories(models_path, spaces_path, predictions_path)
        
        print(f'model repository set to {models_path}')
        print(f'space repository set to {spaces_path}')
        print(f'predictions repository set to {predictions_path}')

        return True, config

    if path is None:  # set default

        # If flame has been already configured, just show values in screen and return values
        if config['config_status'] == True:
            print(f'current model repository is {config["model_repository_path"]}')
            print(f'current space repository is {config["space_repository_path"]}')
            print(f'current predictions repository is {config["predictions_repository_path"]}')

            return True, config

        # If flame has not been already configured assign defaults
        models_path = appdirs.user_data_dir('models', 'flame')
        spaces_path = appdirs.user_data_dir('spaces', 'flame')
        predictions_path = appdirs.user_data_dir('predictions', 'flame')
    else :

        try:
            source_dir = os.path.realpath(path)
        except:
            return False, f'input path {path} is not recognized as a valid path'

        models_path = os.path.join (source_dir,'models')
        predictions_path = os.path.join (source_dir,'predictions')
        spaces_path = os.path.join (source_dir,'spaces')

    # at this point, paths must has been assigned
    print(f'model repository will be set to {models_path}')
    print(f'space repository will be set to {spaces_path}')
    print(f'predictions repository will be set to {predictions_path}')
    print('continue?(y/n)')

    if ask_user():

        try:
            if not os.path.isdir(models_path):
                os.mkdir(models_path)
            if not os.path.isdir(spaces_path):
                os.mkdir(spaces_path)
            if not os.path.isdir(predictions_path):
                os.mkdir(predictions_path)
        except Exception as e:
            return False, e
        
        utils.set_repositories(models_path, spaces_path, predictions_path)

        print(f'model repository set to {models_path}')
        print(f'space repository set to {spaces_path}')
        print(f'predictions repository set to {predictions_path}')
        return True, config

    else:
        print('aborting...')
        return False, 'configuration aborted'
