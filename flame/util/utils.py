#! -*- coding: utf-8 -*-

# Description    Misc tools
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
# Copyright 2018 Manuel Pastor
##
# This file is part of Flame
##
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
##
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import hashlib
import os
import sys
import yaml
import random
import string
import pathlib
import re
import warnings

from flame.util import get_logger

LOG = get_logger(__name__)


def get_conf_yml_path() -> str:
    '''
    recovers the path of the configuration yml file

    Returns:
    --------
    str, path where conf.yaml is

    TODO: be sure that the conf.yaml exists and raise
    err if doesn't
    '''
    # conf is in /flame/flame/conf.yaml
    # __file__ is /flame/flame/util/utils.py
    # jump two parents back with .parents[1]
    source_dir = pathlib.Path(__file__).resolve().parents[1]
    return os.path.join(source_dir, 'config.yaml')


def _read_configuration() -> dict:
    '''
    Reads configuration file "config.yaml" and checks
    sanity of model repository path.


    Returns:
    --------
    dict
    '''
    # LOG.info('reading configuration')
    with open(get_conf_yml_path(), 'r') as config_file:
        conf = yaml.load(config_file)

    model_path = pathlib.Path(conf['model_repository_path'])

    model_abs_path = pathlib.Path(model_path).resolve()

    # LOG.debug(f'changed path from {model_path} to {model_abs_path}')
    conf['model_repository_path'] = str(model_abs_path)
    # LOG.info('Configuration loaded')
    return conf


def check_repository_path() -> None:
    """
    Checks existence of module repository path in config file
    Use only in flame_scr, it uses user input so it's a CLI tools
    """
    LOG.debug('reading configuration')

    config_path = get_conf_yml_path()
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    old_model_path = config['model_repository_path']

    LOG.debug(f'Current model repository path: {old_model_path}')

    model_path = pathlib.Path(old_model_path)
    # check if path exists
    while not model_path.exists():
        LOG.warning(f"Model repository path '{model_path}'"
                    " in config file doesn't exists.")

        print("\nEnter a correct model repository path:")
        user_path = input()
        model_path = pathlib.Path(user_path)

    model_abs_path = str(model_path.resolve())

    # if repo path has been updated
    if old_model_path != model_abs_path:
        LOG.debug('Model repo path changed from '
                  f'{old_model_path} to {model_abs_path}')

        config['model_repository_path'] = model_abs_path

        # write new config to config file
        with open(config_path, 'w') as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

        LOG.info('Model repository path updated')

    # # finds C: or D:
    # rex = re.compile('^.:')
    # match_windows = rex.findall(str(model_path))

    # # extra check if on linux and path starts with char followed by ':'
    # if sys.platform == 'linux' and match_windows:
    #     raise ValueError('Windows path found in config.yml'
    #                      'model repository path:'
    #                      f'"{model_path}".'
    #                      '\nPlease write a correct path manually')


def write_config(config: dict) -> None:
    """Writes the configuration to disk"""
    with open(get_conf_yml_path(), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def set_model_repository(path=None):
    """
    Set the model repository path.
    This is the dir where flame is going to create and load models.

    if path is None, model dir will be set to the default in the
    flame root directory.

    Returns:
    --------
    None
    """
    with open(get_conf_yml_path(), 'r') as f:
        configuration = yaml.load(f)

    if path is None:  # set to default path
        model_root_path = os.path.join(
            pathlib.Path(__file__).resolve().parents[1],
            'models/')
        configuration['model_repository_path'] = str(model_root_path)
    else:
        new_path = pathlib.Path(path)
        configuration['model_repository_path'] = str(new_path.resolve())

    write_config(configuration)


def model_repository_path():
    '''
    Returns the path to the root of the model repository,
    containing all models and versions
    '''
    configuration = _read_configuration()
    return configuration['model_repository_path']


def model_tree_path(model):
    '''
    Returns the path to the model given as argumen, containg all versions
    '''

    return os.path.join(model_repository_path(), model)


def model_path(model, version):
    '''
    Returns the path to the model and version given as arguments
    '''

    modpath = model_tree_path(model)

    if version == 0:
        modpath = os.path.join(modpath, 'dev')
    else:
        modpath = os.path.join(modpath, 'ver%0.6d' % (version))

    return modpath


def module_path(model, version):
    '''
    Returns the path to the model and version given as arguments,
    in Python synthax (separated by ".").

    Also adds the model repository path to the Python path, so the relative
    module path can be understood and the module imported.
    '''

    modreppath = model_repository_path()
    if modreppath not in sys.path:
        sys.path.insert(0, modreppath)

    # print (sys.path)

    # modpath = 'models'+'.'+model
    modpath = model

    if version == 0:
        modpath += '.dev'
    else:
        modpath += '.ver%0.6d' % (version)

    return modpath


def md5sum(filename, blocksize=65536):
    '''
    Returns the MD5 sum of the file given as argument
    '''

    hash = hashlib.md5()

    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)

    return hash.hexdigest()


def intver(raw_version):
    '''
    Returns an int describing at best the model version provided as argument
    '''

    if raw_version is None:
        return 0

    try:
        version = int(raw_version)
    except BaseException:
        version = 0

    return version


def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    '''
    Return a random ID (used for temp files) with uppercase letters and numbers
    '''

    return ''.join(random.choice(chars) for _ in range(size))


def get_parameters(model, version):

    parameters_file_path = model_path(model, version)
    parameters_file_name = os.path.join (parameters_file_path,'parameters.yaml')

    if not os.path.isfile(parameters_file_name):
        return False, None

    try:
        with open(parameters_file_name, 'r') as pfile:
            parameters = yaml.load(pfile)
    except Exception as e:
        return False, e

    parameters['endpoint'] = model
    parameters['version'] = version
    parameters['model_path'] = parameters_file_path
    parameters['md5'] = md5sum(parameters_file_name)

    return True, parameters


def add_result(results, var, _key, _label, _type, _dimension='objs',
               _description=None, _relevance=None):
    '''
    Utility function to insert information within the "result" dictionary, indexing 
    it appropriatelly in the "manifest" and "meta" keys
    
    _key         (str)
                   key for including this data in results dictionary
    
    _label       (str) 
                   descriptive text used to label this data in tables
       
    _type        [label | decoration | smiles | result | confidence | method]
                   cathegory of data, used to decide how showing it in GUI's
    
    _dimension   [single | vars | objs]
                   if the data is one/more isolated values (single) or an array
                   with data for each X variable (vars) or object (objs) 
    
    _description (str)
                    a long human readable description of the information
    
    _relevance   [main | None]
                    the main label is asigned to data to be highlighed 

    '''

    # if the key 'manifest' has not been created already, initiate it with an empty list
    if 'manifest' not in results:
        results['manifest'] = []

    # add the data to results
    results[_key] = var

    # insert the information in manifest
    manifest_item = {'key': _key,
                     'label': _label,
                     'type': _type,
                     'dimension': _dimension,
                     'description': _description,
                     'relevance': _relevance
                     }
    results['manifest'].append(manifest_item)

    # if the are adding data of type 'main' insert the key in meta 
    if _relevance == 'main':
        if 'meta' not in results:
            results['meta'] = {'main': [_key]}
        else:
            results['meta']['main'].append(_key)


def is_empty(mylist):
    for i in mylist:
        if i is not None:
            return False
    return True


def get_sdf_activity_value(mol, parameters: dict) -> float:
    """ Returns the value of the activity present in a SDFIle mol 
    
    The field containing this value is recognized using the parameter 'SDFile_activity"
    If this value is undefined or the field does not exists or is not a float, it returns None

    Returns activity value as float or None
    """

    activity_num = None

    if parameters['SDFile_activity'] is not None:  # if the parameter exists

        if mol.HasProp(parameters['SDFile_activity']):  # if the SDFile contains the field
           
            activity_str = mol.GetProp(parameters['SDFile_activity'])
            try:
                activity_num = float(activity_str) # cast val to float to be sure it is 
            except Exception as e:
                LOG.error('The SDFile activity value cannot be converted'
                            f' to float: {e}')

    return activity_num

def get_sdf_value(mol, value_label) :
    """ Returns the value of the certain field present in a SDFIle mol 
    
    The field containing this value is recognized using the value_label
    If this field does not exists or is not a float, it returns None

    Returns either a float or None
    """

    value_num = None

    # if the SDFile contains the field
    if mol.HasProp(value_label):  

        value_str = mol.GetProp(value_label)
        
        # cast val to float to be sure it is such or return None otherwyse
        try:
            
            value_num = float(value_str)  

        except Exception as e:
            
            LOG.error('An SDFile value cannot be converted'
                        f' to float: {e}')

    return value_num

