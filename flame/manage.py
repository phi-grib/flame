#! -*- coding: utf-8 -*-

# Description    Flame Manage class
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

import os
import sys
import shutil
import tarfile
import json
import pickle
import pathlib
import numpy as np

from flame.util import utils, get_logger
from flame.parameters import Parameters

LOG = get_logger(__name__)


def set_model_repository(path=None):
    """
    Set the model repository path.
    This is the dir where flame is going to create and load models
    """
    utils.set_model_repository(path)

    # this is a console oriented tool which prints messages. Avoid use of LOG.info
    LOG.info(f'Model repository updated to {path}')
    #print(f'Model repository updated to {path}')
    return True, 'model repository updated'


def action_new(model):
    '''
    Create a new model tree, using the given name.
    This creates the development version "dev",
    copying inside default child classes
    '''

    if not model:
        return False, 'empty model label'

    # importlib does not allow using 'test' and issues a misterious error when we
    # try to use this name. This is a simple workaround to prevent creating models 
    # with this name 
    if model == 'test':
        #LOG.warning(f'the name "test" is disallowed, please use any other name')
        return False, 'the name "test" is disallowed, please use any other name'

    # Model directory with /dev (default) level
    ndir = pathlib.Path(utils.model_tree_path(model)) / 'dev'

    # check if there is already a tree for this endpoint
    if ndir.exists():
        #LOG.warning(f'Endpoint {model} already exists')
        return False, f'Endpoint {model} already exists'

    try:
        ndir.mkdir(parents=True)
        LOG.debug(f'{ndir} created')
    except:
        return False, f'Unable to create path for {model} endpoint'

    # Copy classes skeletons to ndir
    wkd = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    children_names = ['apply', 'idata', 'odata', 'learn']

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

    LOG.info(f'New endpoint {model} created')
    #print(f'New endpoint {model} created')
    return True, 'new endpoint '+model+' created'


def action_kill(model):
    '''
    removes the model tree described by the argument
    '''

    if not model:
        return False, 'Empty model name'

    ndir = utils.model_tree_path(model)

    if not os.path.isdir(ndir):
        #LOG.error(f'Model {model} not found')
        return False, f'Model {model} not found'

    try:
        shutil.rmtree(ndir, ignore_errors=True)
    except:
        return False, f'Failed to remove model {model}'

    LOG.info(f'Model {model} removed')
    #print(f'Model {model} removed')
    return True, f'Model {model} removed'


def action_publish(model):
    '''
    clone the development "dev" version as a new model version,
     assigning a sequential version number
    '''

    if not model:
        return False, 'Empty model label'

    base_path = utils.model_tree_path(model)

    if not os.path.isdir(base_path):
        #LOG.error(f'Model {model} not found')
        return False, f'Model {model} not found'

    # gets version number
    v = [int(x[-6:]) for x in os.listdir(base_path) if x.startswith("ver")]

    if not v:
        max_version = 0
    else:
        max_version = max(v)

    new_path = os.path.join(base_path,f'ver{max_version+1:06}')

    if os.path.isdir(new_path):
        #LOG.error(f'Versin {v} of model {model} not found')
        return False, f'Version {max_version+1} of model {model} already exists'

    src_path = os.path.join (base_path,'dev')

    try:
        shutil.copytree(src_path, new_path)
    except:
        return False, f'Unable to copy contents of dev version for model {model}'

    LOG.info(f'New model version created from {src_path} to {new_path}')
    return True, f'New model version created from {src_path} to {new_path}'


def action_remove(model, version):
    '''
    Remove the version indicated as argument from the model tree indicated
    as argument
    '''

    if not model:
        return False, 'Empty model label'

    if version == 0:
        return False, 'Development version cannot be removed, provide a version number'

    rdir = utils.model_path(model, version)
    if not os.path.isdir(rdir):
        return False, f'Version {version} not found'

    shutil.rmtree(rdir, ignore_errors=True)
    LOG.info(f'Version {version} of model {model} has been removed')
    return True, f'Version {version} of model {model} has been removed'


def action_list(model):
    '''
    In no argument is provided lists all models present at the repository 
     otherwyse lists all versions for the model provided as argument
    '''

    # if no model name is provided, just list the model names
    if not model:
        rdir = utils.model_repository_path()

        num_models = 0
        LOG.info('Models found in repository:')
        for x in os.listdir(rdir):
            num_models += 1
            LOG.info('\t'+x)
        LOG.debug(f'Retrieved list of models from {rdir}')
        return True, ''
        return True, f'{num_models} models found'


    # if a model name is provided, list versions
    base_path = utils.model_tree_path(model)

    num_versions = 0
    for x in os.listdir(base_path):
        if x.startswith("ver"):
            num_versions += 1
            LOG.info(f'\t{model} : {x}')

    return True, f'Model {model} has {num_versions} published versions'


def action_import(model):
    '''
    Creates a new model tree from a tarbal file with the name "model.tgz"
    '''

    if not model:
        return False, 'Empty model label'

    # convert model to endpoint string
    base_model = os.path.basename(model)
    endpoint = os.path.splitext(base_model)[0]
    ext = os.path.splitext(base_model)[1]

    base_path = utils.model_tree_path(endpoint)

    if os.path.isdir(base_path):
        return False, f'Endpoint {endpoint} already exists'

    if ext != '.tgz':
        importfile = os.path.abspath(model+'.tgz')
    else:
        importfile = model

    LOG.info(f'Importing {importfile} ...')

    if not os.path.isfile(importfile):
        LOG.info(f'Importing package {importfile} not found')
        return False, f'Importing package {importfile} not found'

    try:
        os.mkdir(base_path)
    except Exception as e:
        return False, f'error creating directory {base_path}: {e}'

    with tarfile.open(importfile, 'r:gz') as tar:
        tar.extractall(base_path)

    LOG.info(f'Endpoint {endpoint} imported OK')
    return True, 'Endpoint '+endpoint+' imported OK'


def action_export(model):
    '''
    Exports the whole model tree indicated in the argument as a single
    tarball file with the same name.
    '''

    if not model:
        return False, 'Empty model label'

    current_path = os.getcwd()
    exportfile = os.path.join(current_path,model+'.tgz')

    base_path = utils.model_tree_path(model)

    if not os.path.isdir(base_path):
        return False, 'Unable to export, endpoint directory not found'

    # change to model repository to tar the file from there
    os.chdir(base_path)

    itemend = os.listdir()
    itemend.sort()

    with tarfile.open(exportfile, 'w:gz') as tar:
        for iversion in itemend:
            if not os.path.isdir(iversion):
                continue
            tar.add(iversion)

    # return to current directory
    os.chdir(current_path)

    LOG.info(f'Model {model} exported as {model}.tgz')
    return True, f'Model {model} exported as {model}.tgz'


# TODO: implement refactoring, starting with simple methods
def action_refactoring(file):
    '''
    NOT IMPLEMENTED,
    call to import externally generated models (eg. in KNIME or R)
    '''

    print('refactoring')

    return True, 'OK'


def action_info(model, version, output='text'):
    '''
    Returns a text or JSON with results info for a given model and version
    '''

    if model is None:
        return False, 'Empty model label'


    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):

        # compatibity method. use info.pkl
        if not os.path.isfile(os.path.join(rdir, 'info.pkl')):
            return False, 'Info file not found'

        with open(os.path.join(rdir, 'info.pkl'), 'rb') as handle:
            #retrieve a pickle file containing the keys 'model_build' 
            #and 'model_validate' of results
            info = pickle.load(handle)
            info += pickle.load(handle)
        # end of compatibility method

    else:
        # new method, use results.pkl
        if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
            return False, 'Info file not found'

        with open(os.path.join(rdir, 'results.pkl'), 'rb') as handle:
            results = pickle.load(handle)
        
        info = None
        if 'model_build_info' in results:
            info =  results['model_build_info']

        if info == None:
            return False, 'Info not found'

        if 'model_valid_info' in results:
            info += results['model_valid_info']
        
        if info == None:
            return False, 'Info not found'

    # when this function is called from the console, output is 'text'
    # write and exit
    if output == 'text':

        LOG.info (f'informing model {model} version {version}')

        for val in info:
            if len(val) < 3:
                LOG.info(val)
            else:
                LOG.info(f'{val[0]} ({val[1]}) : {val[2]}')
        return True, 'model informed OK'

    # this is only reached when this funcion is called from a web service
    # asking for a JSON
    # 
    # this code serializes the results in a list and then converts it 
    # to a JSON  
    json_results = []
    for i in info:
        # results must be checked to avoid numpy elements not JSON serializable
        if 'numpy.int64' in str(type(i[2])):
            try:
                v = int(i[2])
            except Exception as e:
                LOG.error(e)
                v = None
            json_results.append((i[0], i[1], v))

        elif 'numpy.float64' in str(type(i[2])):
            try:
                v = float(i[2])
            except Exception as e:
                LOG.error(e)
                v = None
            json_results.append((i[0], i[1], v))

        elif isinstance(i[2], np.ndarray):
            if 'bool_' in str(type(i[2][0])):
                temp_results = [
                    'True' if x else 'False' for x in i[2]]
            else:
                # This removes NaN and and creates
                # a plain list of formatted floats from ndarrays
                temp_results = [float("{0:.4f}".format(x)) if not np.isnan(x) else None for x in i[2]]

            json_results.append((i[0], i[1], temp_results ))

        else:
            json_results.append(i)

    return True, json.dumps(json_results)


def action_results(model, version=None, ouput_variables=False):
    ''' Returns a JSON with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
        return False, 'results not found'

    # retrieve a pickle file containing the keys 'model_build' 
    # and 'model_validate' of results
    with open(os.path.join(rdir, 'results.pkl'), 'rb') as handle:
        results = pickle.load(handle)

    # this code serializes the results in a list and then converts it 
    # to a JSON  

    json_results = {}

    # creates a list with the keys which should NOT be included
    black_list = []
    for k in results['manifest']:

        ###
        # By default do not include 'var' arrays, only 'obj' arrays
        # to avoid including the X matrix and save space
        # 
        # this black list can be easily tuned to include everything
        # or to remove other keys
        ###
        if not ouput_variables:
            if (k['dimension'] in ['vars']):
                black_list.append(k['key'])

    # iterate keys and for those not in the black list
    # format the information to JSON
    for key in results:

        if key in black_list:
            continue

        value = results[key]

        # np.arrays cannot be serialized to JSON and must be transformed
        if isinstance(value, np.ndarray):

            # do not process bi-dimensional arrays
            if len (np.shape(value)) > 1 :
                continue

            # boolean must be transformed to 'True' or 'False' strings
            if 'bool_' in str(type(value[0])):
                json_results[key] = [
                    'True' if x else 'False' for x in value]

            # we assume that np.array must contain np.floats
            else:
                # This removes NaN and and creates
                # a plain list from ndarrays
                json_results[key] = [x if not np.isnan(
                    x) else None for x in value]

        else:
            json_results[key] = value
        
        try:
            output = json.dumps(json_results)
        except:
            return False, 'Unable to serialize to JSON the model results'

    return True, output


def action_parameters (model, version=None, oformat='text'):
    ''' Returns a JSON with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    param = Parameters()
    param.loadYaml(model, version)

    if oformat == 'JSON':
        return True, param.dumpJSON()

    else:

        order = ['input_type', 'quantitative', 'SDFile_activity', 'SDFile_name', 
        'SDFile_experimental', 'normalize_method', 'ionize_method', 'convert3D_method', 
        'computeMD_method', 'model', 'modelAutoscaling', 'tune', 'conformal', 
        'conformalSignificance', 'ModelValidationCV', 'ModelValidationLC', 
        'ModelValidationN', 'ModelValidationP', 'output_format', 'output_md', 
        'TSV_activity', 'TSV_objnames', 'TSV_varnames', 'imbalance', 
        'feature_selection', 'feature_number', 'mol_batch', 'ext_input', 
        'model_set', 'numCPUs', 'verbose_error', 'modelingToolkit', 
        # 'SVM_parameters','SVM_optimize','RF_parameters', 'RF_optimize', 
        # 'GNB_parameters','PLSR_parameters', 'PLSR_optimize', 
        # 'PLSDA_parameters', 'PLSDA_optimize',
        'endpoint', 'model_path', 
        #'md5', 
        'version']

        for ik in order:
            if ik in param.p:
                k = ik
                v = param.p[k]

                ivalue = ''
                idescr = ''

                if param.extended:
                    if 'value' in v:
                        if not isinstance(v['value'] ,dict):
                            ivalue = v['value']
                        else:
                            ivalue = '*dictionary*'

                    if 'description' in v:
                        idescr = v['description'] 
                else:
                    if not isinstance(v ,dict):
                        ivalue = v
                    else:
                        ivalue = '*dictionary*'

                print (f'{k:30} : {str(ivalue):30} # {idescr}')

        return True, 'parameters listed'


## the following commands are argument-less, intended to be called from a web-service to 
## generate JSON output only

def action_dir():
    '''
    Returns a JSON with the list of models and versions
    '''
    # get de model repo path
    models_path = pathlib.Path(utils.model_repository_path())

    # get directories in model repo path
    dirs = [x for x in models_path.iterdir() if x.is_dir()]

    # if dir contains dev/ -> is model (NAIVE APPROACH)
    # get last dir name [-1]: model name
    model_dirs = [d.parts[-1] for d in dirs if list(d.glob('dev'))]

    results = []
    for imodel in model_dirs:

        # versions = ['dev']
        versions = [{'text': 'dev'}]

        for iversion in os.listdir(utils.model_tree_path(imodel)):
            if iversion.startswith('ver'):
                # versions.append (iversion)
                versions.append({'text': iversion})

        # results.append ((imodel,versions))
        results.append({'text': imodel, 'nodes': versions})

    return True, json.dumps(results)


def action_report():
    '''
    Returns a JSON with the list of models and the results of each one
    '''
    # get de model repo path
    models_path = pathlib.Path(utils.model_repository_path())

    # get directories in model repo path
    dirs = [x for x in models_path.iterdir() if x.is_dir()]

    # if dir contains dev/ -> is model (NAIVE APPROACH)
    # get last dir name [-1]: model name
    model_dirs = [d.parts[-1] for d in dirs if list(d.glob('dev'))]

    results = []
    
    # iterate for models
    for imodel in model_dirs:

        isuccess, ijson = action_info(imodel, 0, output='JSON')

        if not isuccess:
            continue

        iresults = {}
        iresults ['model'] = imodel
        iresults ['results'] = json.loads(ijson)
        results.append(iresults)

    return True, json.dumps(results)