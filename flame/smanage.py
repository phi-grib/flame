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
import json
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
    return True, f'New space version created from {src_path} to {new_path}'


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
    param.loadYaml(space, version, isSpace=True)

    if oformat == 'JSON':
        return True, param.dumpJSON()

    else:

        order = ['input_type', 'quantitative', 'SDFile_activity', 'SDFile_name', 
        'SDFile_experimental', 'normalize_method', 'ionize_method', 'convert3D_method', 
        'computeMD_method', 'model', 'modelAutoscaling', 'tune', 'conformal', 
        'conformalSignificance', 'ModelValidationCV', 'ModelValidationLC', 
        'ModelValidationN', 'ModelValidationP', 'output_format', 'output_md', 
        'TSV_activity', 'TSV_objnames', 'TSV_varnames', 'imbalance', 
        'feature_selection', 'feature_number', 'mol_batch', 
        'ensemble_models', 'ensemble_versions', 'numCPUs', 'verbose_error', 'modelingToolkit', 
        'endpoint', 'model_path', 
        #'md5', 
        'version']

        order += ['MD_settings', 'RF_parameters','RF_optimize',
        'SVM_parameters','SVM_optimize',
        'PLSDA_parameters','PLSDA_optimize',
        'PLSR_parameters','PLSR_optimize',
        'GNB_parameters']


        for ik in order:
            if ik in param.p:
                k = ik
                v = param.p[k]

                ivalue = ''
                idescr = ''
                ioptio = ''

                ## newest parameter formats are extended and contain
                ## rich metainformation for each entry
                if param.extended:
                    if 'value' in v:
                        if not isinstance(v['value'] ,dict):
                            ivalue = v['value']
                        else:
                            # print header of dictionaty
                            print (f'{k} :')

                            # iterate keys assuming existence of value and description
                            for intk in v['value']:
                                intv = v['value'][intk]

                                iivalue = ''
                                if "value" in intv:                                
                                    iivalue = intv["value"]

                                iidescr = ''
                                if "description" in intv and intv["description"] is not None:
                                    iidescr = intv["description"]

                                iioptio = ''
                                if 'options' in intv:
                                    toptio = intv['options']

                                    if isinstance(toptio, list):
                                        if toptio != [None]:
                                            iioptio = f' {toptio}'

                                if isinstance (iivalue, float):
                                    iivalue =  f'{iivalue:f}'
                                elif iivalue is None:
                                    iivalue = ''

                                print (f'   {intk:27} : {str(iivalue):30} #{iioptio} {iidescr}')

                            continue

                    if 'description' in v:
                        idescr = v['description'] 

                    if 'options' in v:
                        toptio = v['options']

                        if isinstance(toptio, list):
                            ioptio = f' {toptio}'

                ### compatibility: old stile parameters
                else:
                    if not isinstance(v ,dict):
                        ivalue = v
                    else:
                        ivalue = '*dictionary*'
                ### end compatibility

                if isinstance (ivalue, float):
                    ivalue =  f'{ivalue:f}'
                elif ivalue is None:
                    ivalue = ''

                print (f'{k:30} : {str(ivalue):30} #{ioptio} {idescr}')

        return True, 'parameters listed'

## the following commands are argument-less, intended to be called from a web-service to 
## generate JSON output only

def action_info(space, version):
    '''
    Returns a text or JSON with results info for a given model and version
    '''

    if space is None:
        return False, 'Empty space label'

    rdir = utils.space_path(space, version)

    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
        return False, 'Info file not found'

    from flame.conveyor import Conveyor

    conveyor = Conveyor()
    with open(os.path.join(rdir, 'results.pkl'), 'rb') as handle:
        conveyor.load(handle)
    
    info =  conveyor.getVal('space_build_info')
    
    if info == None:
        return False, 'Info not found'
   
    # this code serializes the results in a list and then converts it 
    # to a JSON  
    json_results = []
    for i in info:
        json_results.append(conveyor.modelInfoJSON(i))

    #print (json.dumps(json_results))
    return True, json.dumps(json_results)


def action_dir():
    '''
    Returns a JSON with the list of spaces and versions
    '''
    # get de space repo path
    spaces_path = pathlib.Path(utils.space_repository_path())

    # get directories in space repo path
    dirs = [x for x in spaces_path.iterdir() if x.is_dir()]

    # if dir contains dev/ -> is space (NAIVE APPROACH)
    # get last dir name [-1]: space name
    space_dirs = [d.parts[-1] for d in dirs if list(d.glob('dev'))]

    results = []
    for ispace in space_dirs:
        idict = {}
        idict ["spacename"] = ispace
        versions = [0]

        for iversion in os.listdir(utils.space_tree_path(ispace)):
            if iversion.startswith('ver'):
                versions.append(utils.modeldir2ver(iversion))

        idict ["versions"] = versions
        results.append(idict)

    # print (json.dumps(results))
    return True, json.dumps(results)

def action_searches_result (label):
    '''
    try to retrieve the searches result with the label used as argument
    returns 
        - (False, Null) if it there is no directory or the search 
          pickle file cannot be found 
        
        - (True, JSON) with the results otherwyse
    '''
    opath = tempfile.gettempdir()
    if not os.path.isdir(opath):
        return False, f'directory {opath} not found'

    # default in case label was not provided
    if label is None:
        label = 'temp'

    iconveyor = Conveyor()

    search_pkl_path = os.path.join(opath,'similars-'+label+'.pkl')
    if not os.path.isfile(search_pkl_path):
        return False, f'file {search_pkl_path} not found'

    with open(search_pkl_path, 'rb') as handle:
        success, message = iconveyor.load(handle)

    if not success:
        print (f'error reading prediction results with message {message}')
        return False, None

    if not iconveyor.isKey('search_results'):
        return False, 'search results not found'

    results = iconveyor.getVal('search_results')
    names = iconveyor.getVal('obj_nam')
    if iconveyor.isKey('SMILES'):
        smiles = iconveyor.getVal('SMILES')
    if len (results) != len (names):
        return False, 'results length does not match names'

    for i in range (len(results)):
        if iconveyor.isKey('SMILES'):
            print (f'similars to {names[i]} [{smiles[i]}]')
        else:
            print (f'similars to {names[i]}')

        iresult = results[i]
        for j in range (len(iresult['distances'])):
            dist = iresult['distances'][j]
            name = iresult['names'][j]
            smil = iresult['SMILES'][j]
            print (f'   {dist:.3f} : {name} [{smil}]')

    # return a JSON generated by iconveyor
    return True, iconveyor.getJSON()