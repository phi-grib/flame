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


def __read_configuration():
    ''' Reads configuration file "config.yaml". Do not call directly, read configuration variable instead '''
    conf = {}
    source_dir = os.path.dirname(os.path.abspath(__file__))[
        :-5]  # removing '/utils'

    with open(os.path.join(source_dir, 'config.yaml'), 'r') as config_file:
        conf = yaml.load(config_file)

    # if the name of a path starts with '.' we will prepend the path with the source dir
    if conf['model_repository_path'][0] == '.':

        # TODO: I dislike the use of "/" here... but os.path.append does not work well
        conf['model_repository_path'] = source_dir + \
            '/' + conf['model_repository_path'][1:]

    #print (conf)

    return conf


# read configuraton file and store in a variable to prevent reading files more
# than strictly necessary
configuration = __read_configuration()


def model_repository_path():
    ''' Returns the path to the root of the model repository, containing all models and versions '''

    return configuration['model_repository_path']


def model_tree_path(model):
    ''' Returns the path to the model given as argumen, containg all versions '''

    return os.path.join(model_repository_path(), model)


def model_path(model, version):
    ''' Returns the path to the model and version given as arguments '''

    modpath = model_tree_path(model)

    if version == 0:
        modpath = os.path.join(modpath, 'dev')
    else:
        modpath = os.path.join(modpath, 'ver%0.6d' % (version))

    return modpath


def module_path(model, version):
    ''' 

    Returns the path to the model and version given as arguments, in Python synthax (separated by "."). 

    Also adds the model repository path to the Python path, so the relative module path can be 
    understood and the module imported 

    '''

    modreppath = model_repository_path()
    if not modreppath in sys.path:
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
    ''' Returns the MD5 sum of the file given as argument '''

    hash = hashlib.md5()

    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)

    return hash.hexdigest()


def intver(raw_version):
    ''' Returns an int describing at best the model version provided as argument '''

    if raw_version is None:
        return 0

    try:
        version = int(raw_version)
    except:
        version = 0

    return version


def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    ''' Return a random ID (used for temp files) with uppercase letters and numbers '''

    return ''.join(random.choice(chars) for _ in range(size))


def add_result(results, var, _key, _label, _type, _dimension='objs', _description=None, _relevance=None):

    if 'manifest' not in results:
        results['manifest'] = []

    manifest = results['manifest']

    # TODO: check if the _key already exist and add _1 _2 _3 etc as needed

    results[_key] = var

    manifest_item = {'key': _key,                      # key in results
                     'label': _label,                  # descriptive text
                     'type': _type,                    # label, decoration, smiles, result, confidence, method
                     'dimension': _dimension,          # can be single | vars | objs
                     # descriptive text (long)
                     'description': _description,
                     'relevance': _relevance           # main | None
                     }

    manifest.append(manifest_item)

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
