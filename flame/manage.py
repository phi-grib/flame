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
from flame.conveyor import Conveyor
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

    # copy documentation yml file
    documentation_path = wkd / 'children/documentation.yaml'
    shutil.copy(documentation_path, ndir)
  

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


def action_info(model, version, output='text'):
    '''
    Returns a text or an object with results info for a given model and version
    TODO: add Q/C + conf/no-conf + ensem/no-ensem + list of ensemble (when applicable)
    '''

    if model is None:
        return False, 'Empty model label'


    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
        return False, 'Info file not found'

    from flame.conveyor import Conveyor

    conveyor = Conveyor()
    with open(os.path.join(rdir, 'results.pkl'), 'rb') as handle:
        conveyor.load(handle)

    # if there is an error, return the error Message        
    if conveyor.getError():
        error = conveyor.getErrorMessage()
        return False, error

    # collect warnings
    warning_info = None
    
    warning = conveyor.getWarningMessage()
    if warning != None:
        warning_info = [('warning', 'runtime warning', warning)]

    # collect build and validation info
    build_info = conveyor.getVal('model_build_info')
    valid_info = conveyor.getVal('model_valid_info')
    type_info  = conveyor.getVal('model_type_info')

    # merge everything 
    info = None

    for iinfo in (build_info, valid_info, type_info, warning_info):
        if info == None:
            info = iinfo
        else:
            if iinfo != None:
                info+=iinfo

    if info == None:
        return False, 'No relevant information found'

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
    # json_results = []
    # for i in info:
    #     json_results.append(conveyor.modelInfoJSON(i))

    #print (json.dumps(json_results))
    #return True, json.dumps(json_results)
    return True, info

def action_results(model, version=None, ouput_variables=False):
    ''' Returns an object with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
        return False, 'results not found'

    from flame.conveyor import Conveyor

    conveyor = Conveyor()
    with open(os.path.join(rdir, 'results.pkl'), 'rb') as handle:
        conveyor.load(handle)

    # return True, conveyor.getJSON()
    return True, conveyor


def action_parameters(model, version=None, oformat='text'):
    ''' Returns an object with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    from flame.parameters import Parameters

    param = Parameters()
    success, results = param.loadYaml(model, version)

    if not success:
        print (f'error obtaining parametes for model {model} : {results}')
        return False, results

    if oformat == 'JSON':
        # return True, param.dumpJSON()
        return True, param.dumpJSON()

    else:

        order = ['input_type', 'quantitative', 'SDFile_activity', 'SDFile_name', 'SDFile_id',
        'SDFile_experimental', 'SDFile_complementary', 'normalize_method', 'ionize_method', 'convert3D_method', 
        'computeMD_method', 'model', 'modelAutoscaling', 'tune', 'conformal', 
        'conformalSignificance', 'ModelValidationCV', 'ModelValidationLC', 
        'ModelValidationN', 'ModelValidationP', 'output_format', 'output_md', 
        'TSV_activity', 'TSV_objnames', 'TSV_varnames', 'imbalance', 
        'feature_selection', 'feature_number', 'mol_batch',  
        'ensemble_names','ensemble_versions', 'numCPUs', 'verbose_error', 'modelingToolkit', 
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

def action_documentation(model, version=None, doc_file=None, oformat='text'):
    ''' Returns an object with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    from flame.documentation import Documentation
    
    # get de model repo path
    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
        return False, 'Info file not found' 

    doc = Documentation(model, version)

    if doc_file is not None:
        # use the param_file to update existing parameters at the model
        # directory and save changes to make them persistent
        success, message = doc.delta(model, 0, doc_file, iformat='YAML')
    doc = Documentation(model, version)
    if oformat == 'JSON':
        # return True, doc.dumpJSON()
        return True, doc

    else:
        order = ['ID', 'Version', 'Contact', 'Institution', 'Date', 'Endpoint',
         'Endpoint_units', 'Interpretation', 'Dependent_variable', 'Species',
        'Limits_applicability', 'Experimental_protocol', 'Model_availability',
        'Data_info', 'Algorithm', 'Software', 'Descriptors', 'Algorithm_settings',
        'AD_method', 'AD_parameters', 'Goodness_of_fit_statistics', 
        'Internal_validation_1', 'Internal_validation_2', 'External_validation',
        'Comments', 'Other_related_models', 'Date_of_QMRF', 'Data_of_QMRF_updates',
        'QMRF_updates', 'References', 'QMRF_same_models', 'Comment_on_the_endpoint',
        'Endpoint_data_quality_and_variability', 'Descriptor_selection'
        ]


        for ik in order:
            if ik in doc.fields:
                k = ik
                v = doc.fields[k]

                ivalue = ''
                idescr = ''
                ioptio = ''

                ## newest parameter formats are extended and contain
                ## rich metainformation for each entry
                if 'value' in v:
                    if not isinstance(v['value'] ,dict):
                        ivalue = v['value']
                    else:
                        # print header of dictionary
                        print (f'{k} :')

                        # iterate keys assuming existence of value and description
                        for intk in v['value']:
                            intv = v['value'][intk]
                            if not isinstance(intv, dict):
                                print (f'   {intk:27} : {str(intv):30}')  #{iioptio} {iidescr}')
                            
                            else:
                                #print(intk)
                                intv = v['value'][intk]

                                iivalue = ''
                                if "value" in intv:                                
                                    iivalue = intv["value"]
                                # else: 
                                #     iivalue = intv

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

                print (f'{k:30} : {str(ivalue):30} #{ioptio} {idescr}')

        return True, 'parameters listed'

def action_dir():
    '''
    Returns a list of models and versions
    TODO: add action_info for each model
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
        idict = {}
        idict ["modelname"] = imodel
        idict ["version"] = 0
        idict ["info"] = action_info(imodel, 0, output=None)[1]
        results.append(idict)

        for iversion in os.listdir(utils.model_tree_path(imodel)):
            if iversion.startswith('ver'):
                idict = {}
                idict ["modelname"] = imodel
                idict ["version"] = utils.modeldir2ver(iversion)
                idict ["info"] = action_info(imodel, idict ["version"], output=None)[1]
                results.append(idict)

    print (results)
    return True, results


def action_report():
    '''
    Returns a list of models and the results of each one
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
                isuccess, iresult=action_info(imodel_name, iver, output='JSON')
            except:
                continue

            if not isuccess:
                continue
            
            # build a tuple (version, object) for each version and append 
            imodel_vers_info.append((iver, iresult ))

        # build a tuple (model_name, [version_info]) for each model and append
        results.append((imodel_name, imodel_vers_info))
        
    #print (json.dumps(results))
    # return True, json.dumps(results)
    return True, results

def getdate (element):
    return element[0]

def action_predictions_list ():
    '''
    shows a table with the list of predictions 
    '''
    # get de model repo path
    predictions_path = pathlib.Path(utils.predictions_repository_path())

    # get directories in model repo path
    dirs = [x for x in predictions_path.iterdir() if x.is_dir()]

    result = []
    iresult = []
    # iterate models
    for d in dirs:

        #label is retrieved from the directory name
        label = d.parts[-1]

        #metainfo is extracted from prediction-meta picke
        with open(d.joinpath('prediction-meta.pkl'), 'rb') as handle:
            endpoint = pickle.load (handle)
            version  = pickle.load (handle)
            ifile    = pickle.load (handle)
            time     = pickle.load (handle)
            timestamp= pickle.load (handle)

        # ifile is simplified to avoid discossing the repository
        ifile = os.path.basename(ifile)

        # ensemble models are hidden
        if label[0:8]=='ensemble':
            continue

        # add as a tupla 
        iresult.append( ( label, endpoint, version, time, ifile) )

        # format as a text line for reverse date sorting and printing
        line = f'{label:10} {endpoint:15}   {version}   {time}   {ifile}'
        result.append( (timestamp, line) )

    result.sort (reverse=True, key = getdate)

    [print (i[1]) for i in result]

    # return True, json.dumps(jresult)
    return True, iresult

def print_prediction_result (val):
    ''' Prints in the console the content of results given as an 
    argument (val) in a human-readable format 
    '''
    if len(val) < 3:
        print('       ',val)
    else:
        v3 = val[2]
        try:
            v3 = float("{0:.4f}".format(v3))
        except:
            pass

        print(f'       {val[0]} ( {val[1]} ) : {v3}')

def action_predictions_result (label):
    '''
    try to retrieve the prediction result with the label used as argument
    returns 
        - (False, Null) if it there is no directory or the predictions 
          pickle files cannot be found 
        
        - (True, object) with the results otherwyse
    '''
    # get de model repo path
    predictions_path = pathlib.Path(utils.predictions_repository_path())

    label_path = predictions_path.joinpath(label)

    if not label_path.is_dir():
        print (f'directory {label_path} not found')
        return False, None

    result_path = label_path.joinpath('prediction-results.pkl')
    if not result_path.is_file():
        print (f'predictions not found for {label} directory')
        return False, None

    iconveyor = Conveyor()

    with open(result_path, 'rb') as handle:
        success, message = iconveyor.load(handle)

    if not success:
        print (f'error reading prediction results with message {message}')
        return False, None

    # console output    
    print_prediction_result(('obj_num','number of objects',iconveyor.getVal('obj_num')))

    if iconveyor.isKey('external-validation'):
        for val in iconveyor.getVal('external-validation'):
            print_prediction_result (val)   

    if iconveyor.isKey('values'):
        for i in range (iconveyor.getVal('obj_num')):
            print (iconveyor.getVal('obj_nam')[i], '\t', float("{0:.4f}".format(iconveyor.getVal('values')[i])))

    # return iconveyor
    return True, iconveyor

    # return a JSON generated by iconveyor
    # input_type = iconveyor.getMeta('input_type')
    # return True, iconveyor.getJSON(xdata=(input_type == 'model_ensemble'))

def action_predictions_remove (label):
    '''
    try to remove the prediction result with the label used as argument
    returns 
        - (False, message) if it there is no directory or the removal failed 
        - (True, OK) removal succeeded
    '''
    # get de model repo path
    predictions_path = pathlib.Path(utils.predictions_repository_path())

    label_path = predictions_path.joinpath(label)

    if not label_path.is_dir():
        return (False, f'directory {label_path} not found')

    try:
        shutil.rmtree(label_path)
    except Exception as e:
        return (False, f'failed to remove {label_path} with error: {e}')

    return (True, 'OK')


def action_model_template(model, version=None, doc_file=None):
    '''
    Returns a TSV model reporting template
    '''
    from flame.documentation import Documentation
    documentation = Documentation(model, version, context='model')

    if not model:
        return False, 'Empty model label'
    # get de model repo path
    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
        # compatibity method. use info.pkl
        if not os.path.isfile(os.path.join(rdir, 'info.pkl')):
            return False, 'Info file not found'
    else:
        # new method, use results.pkl
        if not os.path.isfile(os.path.join(rdir, 'results.pkl')):
            return False, 'Info file not found'

    if doc_file is not None:
        # use the param_file to update existing parameters at the model
        # directory and save changes to make them persistent
        success, message = documentation.delta(model, 0, doc_file, iformat='YAML')
        print(success, message)

    documentation.get_upf_template2()

    return True, 'Model documentation template created'


def action_prediction_template(model, version=None):
    '''
    Returns a TSV model reporting template
    '''

    from flame.documentation import Documentation

    if not model:
        return False, 'Empty model label'

    documentation = Documentation(model, version, context='prediction')
    documentation.get_prediction_template()

    return True, 'Prediction template created'