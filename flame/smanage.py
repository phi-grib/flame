#! -*- coding: utf-8 -*-

# Description    Flame Manage class
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

import os
import sys
import shutil
import tarfile
import pickle
import pathlib
import tempfile
import numpy as np
from flame.util import utils, get_logger 
from flame.conveyor import Conveyor

LOG = get_logger(__name__)

def set_space_repository(path=None):
    """
    Set the space repository path.
    This is the dir where flame is going to create and load spaces
    """
    utils.set_space_repository(path)

    # this is a console oriented tool which prints messages. Avoid use of LOG.info
    LOG.info(f'space repository updated to {path}')
    #print(f'space repository updated to {path}')
    return True, 'space repository updated'


def action_new(space):
    '''
    Create a new space tree, using the given name.
    This creates the development version "dev",
    copying inside default child classes
    '''

    if not space:
        return False, 'empty space label'

    # importlib does not allow using 'test' and issues a misterious error when we
    # try to use this name. This is a simple workaround to prevent creating spaces 
    # with this name 
    if space == 'test':
        #LOG.warning(f'the name "test" is disallowed, please use any other name')
        return False, 'the name "test" is disallowed, please use any other name'

    # space directory with /dev (default) level
    ndir = pathlib.Path(utils.space_tree_path(space)) / 'dev'

    # check if there is already a tree for this endpoint
    if ndir.exists():
        #LOG.warning(f'Endpoint {space} already exists')
        return False, f'Endpoint {space} already exists'

    try:
        ndir.mkdir(parents=True)
        LOG.debug(f'{ndir} created')
    except:
        return False, f'Unable to create path for {space} endpoint'

    # Copy classes skeletons to ndir
    wkd = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    children_names = ['sapply', 'idata', 'odata', 'slearn']

    for cname in children_names:
        filename = cname + '_child.py'
        src_path = wkd / 'children' / filename
        dst_path = ndir / filename
        try:
            shutil.copy(src_path, dst_path)
        except:
            return False, f'Unable to copy {cname} file'

    LOG.debug(f'copied class skeletons from {src_path} to {dst_path}')
    
    # copy parameter yml file
    params_path = wkd / 'children/parameters.yaml'
    shutil.copy(params_path, ndir)

    # copy documentation yml file
    documentation_path = wkd / 'children/documentation.yaml'
    shutil.copy(documentation_path, ndir)
  
    LOG.info(f'New space {space} created')
    
    return True, 'New space '+space+' created'


def action_kill(space):
    '''
    removes the space tree described by the argument
    '''

    if not space:
        return False, 'Empty space name'

    ndir = utils.space_tree_path(space)

    if not os.path.isdir(ndir):
        #LOG.error(f'space {space} not found')
        return False, f'space {space} not found'

    try:
        shutil.rmtree(ndir, ignore_errors=True)
    except:
        return False, f'Failed to remove space {space}'

    LOG.info(f'space {space} removed')
    #print(f'space {space} removed')
    return True, f'space {space} removed'


def action_publish(space):
    '''
    clone the development "dev" version as a new space version,
     assigning a sequential version number
    '''

    if not space:
        return False, 'Empty space label'

    base_path = utils.space_tree_path(space)

    if not os.path.isdir(base_path):
        #LOG.error(f'space {space} not found')
        return False, f'space {space} not found'

    # gets version number
    v = [int(x[-6:]) for x in os.listdir(base_path) if x.startswith("ver")]

    if not v:
        max_version = 0
    else:
        max_version = max(v)

    new_path = os.path.join(base_path,f'ver{max_version+1:06}')

    if os.path.isdir(new_path):
        #LOG.error(f'Versin {v} of space {space} not found')
        return False, f'Version {max_version+1} of space {space} already exists'

    src_path = os.path.join (base_path,'dev')

    try:
        shutil.copytree(src_path, new_path)
    except:
        return False, f'Unable to copy contents of dev version for space {space}'

    LOG.info(f'New space version created from {src_path} to {new_path}')
    return True, max_version+1


def action_remove(space, version):
    '''
    Remove the version indicated as argument from the space tree indicated
    as argument
    '''

    if not space:
        return False, 'Empty space label'

    if version == 0:
        return False, 'Development version cannot be removed, provide a version number'

    rdir = utils.space_path(space, version)
    if not os.path.isdir(rdir):
        return False, f'Version {version} not found'

    shutil.rmtree(rdir, ignore_errors=True)
    LOG.info(f'Version {version} of space {space} has been removed')
    return True, f'Version {version} of space {space} has been removed'

def action_list(space):
    '''
    Lists all versions for the space provided as argument
    '''

    # if a space name is provided, list versions
    base_path = utils.space_tree_path(space)

    num_versions = 0
    for x in os.listdir(base_path):
        if x.startswith("ver"):
            num_versions += 1
            LOG.info(f'\t{space} : {x}')

    return True, f'space {space} has {num_versions} published versions'

def action_parameters(space, version=None, oformat='text'):
    ''' Returns a JSON with whole results info for a given space and version '''

    if space is None:
        return False, 'Empty space label'

    from flame.parameters import Parameters

    param = Parameters()
    success, results  = param.loadYaml(space, version, isSpace=True)

    if not success:
        print (f'error obtaining parametes for space {space} : {results}')
        return False, results

    if oformat != 'text':
        return True, param

    else:
        yaml = param.dumpYAML()
        for line in yaml:
            print (line)

        return True, 'parameters listed'


## the following commands are argument-less, intended to be called from a web-service to 
## generate JSON output only

def action_info(space, version, output='text'):
    '''
    Returns a text or JSON with results info for a given model and version
    '''

    if space is None:
        if output == 'JSON':
            return False, {'code':1, 'message': 'Empty space label'}
        return False, 'Empty space label'

    meta_path = utils.space_path(space, version)
    meta_file = os.path.join(meta_path, 'space-meta.pkl')
    
    if not os.path.isfile(meta_file):
        if output == 'JSON':
            return False, {'code':0, 'message': 'Info file not found'}
        return False, 'Info file not found'

    with open(meta_file, 'rb') as handle:
        modelID = pickle.load(handle)
        errorMessage = pickle.load(handle)
        warningMessage = pickle.load(handle)
        space_info = pickle.load(handle)
    
    if errorMessage is not None:
        if output == 'JSON':
            return False, {'code':1, 'message': errorMessage}
        return False, 'No relevant information found'
   
    warning_info = None
    if warningMessage is not None:
        warning_info = [('warning', 'runtime warning', warningMessage)]

    info = None
    
    for iinfo in (space_info, warning_info, [('modelID','unique model ID', modelID)]):
        if info == None:
            info = iinfo
        else:
            if iinfo != None:
                info+=iinfo

    if info == None:
        if output == 'JSON':
            return False, {'code':1, 'message': 'No relevant information found'}
        return False, 'No relevant information found'

    if output == 'text':

        LOG.info (f'informing space {space} version {version}')

        for val in info:
            if len(val) < 3:
                LOG.info(val)
            else:
                LOG.info(f'{val[0]} ({val[1]}) : {val[2]}')
        return True, 'space informed OK'

    return True, info


def action_dir():
    '''
    Returns a the list of spaces and versions
    '''
    # get de space repo path
    spaces_path = pathlib.Path(utils.space_repository_path())
    if spaces_path.is_dir() is False:
        return False, 'the spaces repository path does not exist. Please run "flame -c config".'

    # get directories in space repo path
    dirs = [x for x in spaces_path.iterdir() if x.is_dir()]

    # if dir contains dev/ -> is space (NAIVE APPROACH)
    # get last dir name [-1]: space name
    space_dirs = [d.parts[-1] for d in dirs if list(d.glob('dev'))]

    # results = []
    # for ispace in space_dirs:
    #     idict = {}
    #     idict ["spacename"] = ispace
    #     versions = [0]

    #     for iversion in os.listdir(utils.space_tree_path(ispace)):
    #         if iversion.startswith('ver'):
    #             versions.append(utils.modeldir2ver(iversion))

    #     idict ["versions"] = versions
    #     results.append(idict)

    results = []
    for ispace in space_dirs:
        idict = {}
        idict ["spacename"] = ispace
        idict ["version"] = 0
        idict ["info"] = action_info(ispace, 0, output=None)[1]

        results.append(idict)

        for iversion in os.listdir(utils.space_tree_path(ispace)):
            if iversion.startswith('ver'):
                idict = {}
                idict ["spacename"] = ispace
                idict ["version"] = utils.modeldir2ver(iversion)
                idict ["info"] = action_info(ispace, idict ["version"], output=None)[1]

                results.append(idict)


    # print (json.dumps(results))
    return True, results

def action_searches_result (label, output='text'):
    '''
    try to retrieve the searches result with the label used as argument
    returns 
        - (False, Null) if it there is no directory or the search 
          pickle file cannot be found 
        
        - (True, JSON) with the results otherwyse
    '''

    opath = tempfile.gettempdir()
    if not os.path.isdir(opath):
        if output == 'JSON':
            return False, {'code':1, 'message': f'directory {opath} not found'}
        print (f'directory {opath} not found')
        return False, None

    # default in case label was not provided
    if label is None:
        label = 'temp'

    iconveyor = Conveyor()

    search_pkl_path = os.path.join(opath,'similars-'+label+'.pkl')
    if not os.path.isfile(search_pkl_path):

        if output == 'JSON':
            return False, {'code':0, 'message': f'predictions not found for {label} directory'}
        print (f'predictions not found for {label} directory')
        return False, f'file {search_pkl_path} not found'

    with open(search_pkl_path, 'rb') as handle:
        success, message = iconveyor.load(handle)

    if not success:
        if output == 'JSON':
            return False, {'code':1, 'message': f'error reading search results with message {message}'}
        print (f'error reading search results with message {message}')
        return False, None

    if not iconveyor.isKey('search_results'):
        if output == 'JSON':
            return False, {'code':1, 'message': 'search results not found'}
        return False, 'search results not found'

    results = iconveyor.getVal('search_results')
    names = iconveyor.getVal('obj_nam')
    if iconveyor.isKey('SMILES'):
        smiles = iconveyor.getVal('SMILES')
    if len (results) != len (names):
        if output == 'JSON':
            return False, {'code':1, 'message': 'results length does not match names'}
        return False, 'results length does not match names'

    for i in range (len(results)):
        if iconveyor.isKey('SMILES'):
            print (f'similars to {names[i]} [{smiles[i]}]')
        else:
            print (f'similars to {names[i]}')

        iresult = results[i]
        for j in range (len(iresult['distances'])):
            dist = iresult['distances'][j]

            if 'obj_name' in iresult:
                name = iresult['obj_nam'][j]
            else:
                name = '-'
            if 'SMILES' in iresult:
                smil = iresult['SMILES'][j]
            else:
                smil = '-'
            
            if 'obj_id' in iresult:
                idv = iresult['obj_id'][j]
            else:
                idv ='-'
            
            if 'ymatrix' in iresult:
                act = iresult['ymatrix'][j]
            else:
                act = '-'


            print (f'   {dist:.3f} : {name} {idv} {act} [{smil}]')

    # return a JSON generated by iconveyor
    return True, iconveyor