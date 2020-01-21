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


def config(path: str=None) -> bool:
    """Configures model repository.

    Loads config.yaml and writes a correct model repository path
    with the path provided by the user or a default from appdirs
    if the path is not provided.
    """

    # ---- CLI interface -----

    if path is None:  # set default
        default_models_path = Path(appdirs.user_data_dir('models', 'flame'))
        default_spaces_path = Path(appdirs.user_data_dir('spaces', 'flame'))
        default_predictions_path = Path(appdirs.user_data_dir('predictions', 'flame'))

        print(f'Setting model, space and predictions repositories (default) to {default_models_path}, {default_spaces_path} and {default_predictions_path}'
              '\nWould you like to continue?(y/n)')

        if ask_user():
            if default_models_path.exists() or default_spaces_path.exists() or default_predictions_path.exists():
                print(f'These paths already exists. '
                      'Would you like to set them anyway?(y/n)')
                if ask_user():
                    utils.set_repositories(default_models_path, default_spaces_path, default_predictions_path)
                else:
                    print('aborting...')
                    return False

            else:  # models_path doesn't exists
                default_models_path.mkdir(parents=True)
                default_spaces_path.mkdir(parents=True)
                default_predictions_path.mkdir(parents=True)
                utils.set_repositories(default_models_path, default_spaces_path, default_predictions_path)

            print(f'model repository set to {default_models_path}')
            print(f'space repository set to {default_spaces_path}')
            print(f'predictions repository set to {default_predictions_path}')
            return True

        else:
            print('aborting...')
            return False

    else:  # path input by user
        in_path = Path(path).expanduser()
        in_path_models = Path.joinpath(in_path,'models')
        in_path_spaces = Path.joinpath(in_path,'spaces')
        in_path_predictions = Path.joinpath(in_path,'predictions')
        current_models_path = Path(utils.model_repository_path())
        current_spaces_path = Path(utils.space_repository_path())
        current_predictions_path = Path(utils.predictions_repository_path())

        if in_path_models == current_models_path and in_path_spaces == current_spaces_path and in_path_predictions == current_predictions_path:
            print(f'{in_path_models} already is model repository path')
            print(f'{in_path_spaces} already is space repository path')
            print(f'{in_path_predictions} already is predictions repository path')
            return False

        elif not (in_path_models.exists() and in_path_spaces.exists() and in_path_predictions.exists()):
            print("paths doesn't exists. Would you like to create it?(y/n)")

            if ask_user():
                if not in_path_models.exists():
                    in_path_models.mkdir(parents=True)
                if not in_path_spaces.exists():                
                    in_path_spaces.mkdir(parents=True)
                if not in_path_predictions.exists():                
                    in_path_predictions.mkdir(parents=True)
                utils.set_repositories(in_path_models, in_path_spaces, in_path_predictions)
            else:
                print('aborting...')
                return False

        else:  # in_path exists
            utils.set_repositories(in_path_models, in_path_spaces, in_path_predictions)

        print(f'space repository set to {in_path_spaces}')
        print(f'model repository set to {in_path_models}')
        print(f'predictions repository set to {in_path_predictions}')
        return True
