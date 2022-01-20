#! -*- coding: utf-8 -*-

# Description    Misc tools
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

__modules__ = None

import os 
from dotenv import load_dotenv
import sys
import yaml
import random
import pickle
import string
import hashlib
import pathlib
import numpy as np
import codecs
import string
import re 

from flame.util import get_logger
from knowledgehub.api import KnowledgeHubAPI

load_dotenv()

LOG = get_logger(__name__)

def isSingleThread ():
    success, config = read_config()
    if success:
        if 'single_thread' in config:
            return config['single_thread']
    return False

def read_config():
    '''
    Reads configuration file "config.yaml" and checks
    sanity of model repository path.

    Returns:
    --------
    Boolean, dict
    '''

    if 'flame_configuration' in globals():
        return True, globals()['flame_configuration']

    try:
        source_dir = os.path.dirname(os.path.dirname(__file__)) 
        config_nam = os.path.join(source_dir,'config.yaml')
        with open(config_nam,'r') as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        return False, e

    if conf is None:
        return False, 'unable to obtain configuration file'

    if conf['config_status']:
        items = ['model_repository_path', 'space_repository_path', 'predictions_repository_path']
        for i in items:
            try:
                conf[i] = os.path.abspath(conf[i])
            except:
                return False, f'Configuration file incorrect. Unable to convert "{conf[i]}" to a valid path.'
        
    globals()['flame_configuration'] = conf
    return True, conf

def config_test() -> None:
    """ Checks if flame has been configured
     reading the config.yml and checking the config_status flag
    """
    success, config = read_config()

    if success:
        if isinstance(config['config_status'], bool):
            if config['config_status']:
                if os.path.isdir(config['model_repository_path']):
                    return

    sys.exit()  # force exit

def write_config(config: dict) -> None:
    """Writes the configuration to disk"""
    config['config_status'] = True
    
    globals()['flame_configuration'] = config

    source_dir = os.path.dirname(os.path.dirname(__file__)) 
    with open(os.path.join(source_dir,'config.yaml'), 'w') as f:
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
    # source_dir = os.path.dirname(os.path.dirname(__file__)) 
    # with open(os.path.join(source_dir,'config.yaml'),'r') as f:
    #     configuration = yaml.safe_load(f)
    
    success, configuration = read_config()

    if success:
        if path is None:  # set to default path
            model_root_path = os.path.join(
                pathlib.Path(__file__).resolve().parents[1],
                'models/')
            configuration['model_repository_path'] = str(model_root_path)
        else:
            new_path = pathlib.Path(path)
            configuration['model_repository_path'] = str(new_path.resolve())

        write_config(configuration)

# def set_repositories(model_path, space_path, predictions_path):
def set_repositories(root_path, username=None , project=None):
    """
    Set the model repository path.
    This is the dir where flame is going to create and load models.
    Returns:
    --------
    None
    """

    # source_dir = os.path.dirname(os.path.dirname(__file__)) 
    # with open(os.path.join(source_dir,'config.yaml'), 'r') as f:
    #     configuration = yaml.safe_load(f)

    success, configuration = read_config()
    
    if not success:
        return False

    if username is not None:
        root_path = os.path.join(root_path,username)
        try:
            if not os.path.isdir(root_path):
                os.mkdir(root_path)
        except Exception as e:
            print (f'Error {e}')
            return False

    if project is not None:
        root_path = os.path.join(root_path,project)
        try:
            if not os.path.isdir(root_path):
                os.mkdir(root_path)
        except Exception as e:
            print (f'Error {e}')
            return False

    configuration['model_repository_path'] = os.path.join(root_path,'models')
    configuration['space_repository_path'] = os.path.join(root_path,'spaces')
    configuration['predictions_repository_path'] = os.path.join(root_path,'predictions')

    for i in ['model_repository_path', 'space_repository_path', 'predictions_repository_path']:
        path = configuration[i]
        try:
            if not os.path.isdir(path):
                os.mkdir(path)
            print (f'{i}: {configuration[i]}')
        except Exception as e:
            print (f'Error {e}')
            return False

    write_config(configuration)

    return True


def path_expand (path, version):
    ''' 
    Expands the path as required for the version provided as argument 
    '''
    if version == 0:
        return os.path.join(path, 'dev')
    else:
        return os.path.join(path, 'ver%0.6d' % (version))

def model_repository_path():
    '''
    Returns the path to the root of the model repository,
    containing all models and versions
    '''
    success, config = read_config()
    if success: 
        return config['model_repository_path']

def model_tree_path(model):
    '''
    Returns the path to the model given as argumen, containg all versions
    '''
    return os.path.join(model_repository_path(), model)

def model_path(model, version):
    '''
    Returns the path to the model and version given as arguments
    '''
    return path_expand (model_tree_path(model), version)

def getModelID (model, version, object_type='model'):
    path = model_path(model, version)
    meta = os.path.join(path, object_type+'-meta.pkl')

    try:
        with open(meta, 'rb') as handle:
            modelID = pickle.load(handle)
        return True, modelID
    except:
        return False, f'Unable to load modelID from {meta}'

def space_repository_path():
    '''
    Returns the path to the root of the spaces repository,
    containing all models and versions
    '''
    success, config = read_config()
    if success:
        return config['space_repository_path']

def space_tree_path(space):
    '''
    Returns the path to the space given as argumen, containg all versions
    '''
    return os.path.join(space_repository_path(), space)


def space_path(space, version):
    '''
    Returns the path to the model and version given as arguments
    '''
    return path_expand (space_tree_path(space), version)

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

    modpath = model

    if version == 0:
        modpath += '.dev'
    else:
        modpath += '.ver%0.6d' % (version)

    return modpath

def smodule_path(space, version):
    '''
    Returns the path to the space and version given as arguments,
    in Python synthax (separated by ".").

    Also adds the space repository path to the Python path, so the relative
    module path can be understood and the module imported.
    '''
    modreppath = space_repository_path()
    if modreppath not in sys.path:
        sys.path.insert(0, modreppath)

    modpath = space

    if version == 0:
        modpath += '.dev'
    else:
        modpath += '.ver%0.6d' % (version)

    return modpath

def predictions_repository_path():
    '''
    Returns the path to the root of the predictions repository,
    containing all predictions
    '''
    success, config = read_config()
    if success:
        return config['predictions_repository_path']

def safe_copy (inputfile, outputfile):
    ''' this function makes sure that the input file contains only printable chars
        RDKit is very sensitive to the presence of non utf-8 chars and for this reason
        this pre-filter is needed
    '''

    characters_to_keep = string.printable #printable us-ascii only
    search_regex = re.compile("[^%s]" % (re.escape(characters_to_keep)))

    read_stream  = codecs.open(inputfile ,'r',encoding='utf-8', errors='ignore') 
    write_stream = codecs.open(outputfile,'w',encoding='utf-8', errors='ignore')
 
    buffer = 'start'                                                        
    buffer_size = 512*1024 # size in bytes. -1 for loading whole file in 

    while  buffer: # empty string evaluates as False. Any other string as True.
        buffer = read_stream.read(buffer_size)
        write_stream.write(search_regex.sub('?', buffer))

    read_stream.close()
    write_stream.close()

def md5sum(filename, blocksize=65536):
    '''
    Returns the MD5 sum of the file given as argument
    '''
    if filename is None:
        return 0

    if not os.path.isfile(filename):
        return 0
        
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

def modeldir2ver (modeldir):
    '''
    The argument is the name of the directory hosting a
    model version (e.g. '/dev' or '/ver00007'). This function tries to 
    convert it to an integer
    '''
    if modeldir == 'dev':
        return 0
    try:
        version = int(modeldir[-6:])
    except:
        version = 0
    return version

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    '''
    Return a random ID (used for temp files) with uppercase letters and numbers
    '''
    return ''.join(random.choice(chars) for _ in range(size))

def is_empty(mylist):
    ''' returns True if every element in the list is None '''
    for i in mylist:
        if i is not None:
            return False
    return True

def is_string_empty(mylist):
    ''' returns True if every element in the list is None '''
    for i in mylist:
        if i != '':
            return False
    return True

def qualitative_Y (Y):
    ''' Checks if the Y nparray provided as an argument contains only 1 and 0 values and 
        is therefore suitable for being used in qualitative models
    '''
    neg = 0
    pos = 0
    nan = 0
    ext = 0
    for y in Y:
        if np.isclose(y, 0.0):
            neg+=1
        elif np.isclose (y, 1.0):
            pos+=1
        elif np.isnan(y):
            nan+=1 
        else:
            ext+=1

    LOG.debug (f'Y analized. Found {neg} negative, {pos} positive, {nan} NaN and {ext} others objects')

    if neg == 0 or pos == 0:
        return False, f'Y values not suitable for building a qualitative model. Found {neg} negative and {pos} positive objects'

    if ext > 0:
        return False, f'Y values not suitable for building a qualitative model. Found {ext} objects not 1.000 or 0.000'
    
    return True, 'OK'

def module_versions ():
    ''' gather the version of key libraries used for generating the models '''

    from rdkit import __version__ as rdkit_ver
    from sklearn import __version__ as sklearn_ver
    from xgboost import __version__ as xgboost_ver
    from nonconformist import __version__ as nonconformist_ver
    from flame import __version__ as flame_ver

    return {'rdkit':rdkit_ver, 'sklearn':sklearn_ver, 'xgboost':xgboost_ver, 'nonconformist':nonconformist_ver, 'flame': flame_ver}

def compatible_modules (ext_libraries):
    ''' compares a set of library versions (typically retrieved for a stored estimator) with current library versions '''

    int_libraries = module_versions()

    for ilib in int_libraries:
    
        # if a current library is not included in the external skip the comparison
        if ilib not in ext_libraries:
            continue
            # return False, f'missing library "{ilib}"'
    
        # if any versions dont match return false
        #TODO: include a more smart set of rules to prevent warnings with minor release updates 
        if int_libraries[ilib] != ext_libraries[ilib]:
            return False, f'internal library "{ilib}:{int_libraries[ilib]}" '\
                          f'does not match imported library "{ilib}:{ext_libraries[ilib]}"'
    
    return True, 'OK'

def isFingerprint (md):
    fplist = ['morganFP','rdkFP', 'substructureFP']
    
    # if input is a list check every element
    if isinstance(md, list):
        for imd in md:
            if imd in fplist:
                return True
    # if input is a value, just check that is not in the list of fingerprints
    else:
        if md in fplist:
            return True
    
    return False


def connect_api():
    
    api = KnowledgeHubAPI(server='TEST', client_secret=os.getenv('CLIENT_SECRET'))
    api.login(os.getenv("USER_TEST"),os.getenv('PSWD_TEST'))

    return api



def getSmilesByAPI(api,name):
    
    try:
        smiles = api.ChemistryService().getCompoundByName(name)
    except:
        smiles = 0
            
    return smiles
        
     
