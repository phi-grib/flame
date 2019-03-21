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
import numpy as np

from flame.util import utils, get_logger 
# from flame.parameters import Parameters
# from flame.conveyor import Conveyor

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
            xpath = os.path.join(rdir,x) 
            # discard if the item is not a directory
            if not os.path.isdir(xpath):
                continue
            # discard if the directory does not contain a 'dev' directory inside
            if not os.path.isdir(os.path.join(xpath,'dev')):
                continue
            num_models += 1
            LOG.info('\t'+x)
        LOG.debug(f'Retrieved list of models from {rdir}')
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

        from flame.conveyor import Conveyor

        conveyor = Conveyor()
        with open(os.path.join(rdir, 'results.pkl'), 'rb') as handle:
            conveyor.load(handle)
        
        info =  conveyor.getVal('model_build_info')
        info += conveyor.getVal('model_valid_info')
        
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
    
    # this code serializes the results in a list and then converts it 
    # to a JSON  
    json_results = []
    for i in info:
        json_results.append(conveyor.modelInfoJSON(i))

    #print (json.dumps(json_results))
    return True, json.dumps(json_results)


def action_results(model, version=None, ouput_variables=False):
    ''' Returns a JSON with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
        return False, 'results not found'

    from flame.conveyor import Conveyor

    conveyor = Conveyor()
    with open(os.path.join(rdir, 'results.pkl'), 'rb') as handle:
        conveyor.load(handle)

    return True, conveyor.getJSON()


def action_parameters (model, version=None, oformat='text'):
    ''' Returns a JSON with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    from flame.parameters import Parameters

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
        'endpoint', 'model_path', 
        #'md5', 
        'version']

        order += ['MD_settings', 'RF_parameters','RF_optimize',
        'SVM_parameters','SVM_optimize',
        'PLSDA_parameters','PLSDA_optimize',
        'PLSR_parameters','PLSR_optimize',
        'GNB_parameters']

        # if param.extended:
        #     if 'RF' in param.p['model']['value']:
        #         order+=['RF_parameters','RF_optimize']
        #     elif 'SVM' in param.p['model']['value']:
        #         order+=['SVM_parameters','SVM_optimize']
        #     elif 'PLSDA' in param.p['model']['value']:
        #         order+=['PLSDA_parameters','PLSDA_optimize']
        #     elif 'PLSR' in param.p['model']['value']:
        #         order+=['PLSR_parameters','PLSR_optimize']
        #     elif 'GNB' in param.p['model']['value']:
        #         order+='GNB_parameters'

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
                            #print (f'{k:30} :')
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

                                #print (f'   {intk:27} : {str(iivalue):30} #{iioptio} {iidescr}')
                                print (f'   {intk:27} : {str(iivalue):30} ')

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

                #print (f'{k:30} : {str(ivalue):30} #{ioptio} {idescr}')
                print (f'{k:30} : {str(ivalue):30}')

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

    # # if dir contains dev/ -> is model (NAIVE APPROACH)
    # # get last dir name [-1]: model name
    # model_dirs = [d.parts[-1] for d in dirs if list(d.glob('dev'))]

    results = []

    # iterate models
    for d in dirs:
        imodel_name = d.parts[-1]
        imodel_vers = [x.parts[-1] for x in d.iterdir() if x.is_dir()]
        
        # make sure the model contains 'dev' to recognize models
        if 'dev' not in imodel_vers:
            continue
        
        imodel_vers_info = []
        for ivtag in imodel_vers:

            iver = utils.modeldir2ver(ivtag)

            # now we have the model name and version, try to get the ijson text
            try:
                isuccess, ijson = action_info(imodel_name, iver, output='JSON')
            except:
                continue

            if not isuccess:
                continue
            
            # build a tuple (version, JSON) for each version and append 
            imodel_vers_info.append((iver, json.loads(ijson) ))

        # build a tuple (model_name, [version_info]) for each model and append
        results.append((imodel_name, imodel_vers_info))
        
    print (json.dumps(results))
    return True, json.dumps(results)