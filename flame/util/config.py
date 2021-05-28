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


# def ask_user ():
#     ''' utility function to obtain binary confirmation (yes/no) from the user
#     '''
#     userinput = input()

#     if userinput.lower() not in ['yes', 'no', 'y', 'n']:
#         print('Please write "yes", "no", "y" or "n"')
#         return False
#     elif userinput.lower() in ['yes', 'y']:
#         return True
#     return False

# def configure(path: None, silent: False, username='default', project='default'):
def configure(path: None, silent: False, username=None, project=None):
    """Configures model repository.

    Loads config.yaml and writes a correct model repository path
    with the path provided by the user or a default from appdirs
    if the path is not provided.
    """
    
    success, config = utils.read_config()
    source_dir = ''
    if not success:
        return False, config

    ########################################################
    ###  Silent
    ########################################################

    if silent:
        if path is not None:  
            source_dir = os.path.abspath(path)
        else:
            source_dir = os.path.dirname(os.path.dirname(__file__)) 

        success = utils.set_repositories(source_dir, username, project)
        
        if success:
            return True, config
        else:
            return False, 'error setting the repositories'

    ########################################################
    ###  Path not provided
    ########################################################
    if path is None:  # set default

        # If flame has been already configured, just show values in screen and return values
        if config['config_status'] == True:
            for i in ['model_repository_path', 'space_repository_path', 'predictions_repository_path']:
                print (f'{i}: {config[i]}')
            return True, config

        # Assign defaults
        source_dir = appdirs.user_data_dir('flame',False)
        if not os.path.isdir(source_dir):
            try:
                os.mkdir(source_dir)
            except Exception as e:
                return False, f'Error {e}'

    ########################################################
    ###  Path provided
    ########################################################
    else :
        try:
            source_dir = os.path.realpath(path)
        except:
            return False, f'input path {path} is not recognized as a valid path'

    ########################################################
    ###  Common
    ########################################################
    # print(f'root repository will be set to {source_dir}')
    # print('continue?(y/n)')

    # if ask_user():
    success = utils.set_repositories(source_dir, username, project)
    if success:
        return True, config
    else:
        return False, 'error setting the repositories'

    # else:
    #     print('aborting...')
    #     return False, 'configuration aborted'
