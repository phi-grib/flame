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
# import time
import shutil
import tarfile
import pickle
import yaml
import json
import pathlib
import tempfile
# import numpy as np
from flame.util import utils, get_logger 
from flame.conveyor import Conveyor

LOG = get_logger(__name__)

def set_model_repository(path=None):
    """
    Set the model repository path.
    This is the dir where flame is going to create and load models
    """
    utils.set_model_repository(path)

    LOG.info(f'Model repository updated to {path}')
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
    children_names = ['apply', 'idata', 'odata', 'learn', 'slearn', 'sapply']

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
  
    # create default labels
    p = { 'maturity' : 'dev', 'type' : 'unk',
          'subtype' : 'unk', 'endpoint' : 'unk', 'species' : 'unk' }
          
    with open(os.path.join(ndir, 'model-labels.pkl'), 'wb') as fo:
        pickle.dump(p, fo)

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
        if os.path.isdir(rdir) is False:
            return False, 'the models repository path does not exist. Please run "flame -c config".'

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
    import re

    if not model:
        return False, 'Empty model label'

    # convert model to endpoint string
    base_model = os.path.basename(model)
    endpoint = os.path.splitext(base_model)[0]

    # find version in case of single version exports
    version = None
    if re.match("_v[0-9]{6}", endpoint[-8:]):
        version = int(endpoint[-6:])
        endpoint = endpoint[:-8]
    
    ext = os.path.splitext(base_model)[1]
    base_path = utils.model_tree_path(endpoint)

    # safety checks
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

    confidential = False

    # create directory
    try:
        os.mkdir(base_path)
    except Exception as e:
        return False, f'Error creating directory {base_path}: {e}'

    # unpack tar.gz. This is done for any kind of export file
    with tarfile.open(importfile, 'r:gz') as tar:
        tar.extractall(base_path)

    # when importing a single version we need to clone the last folder to 'dev'
    inner_dirs = os.listdir(base_path)
    if not 'dev' in inner_dirs and version is not None:
        
        # assign single version using file name
        version_dir = f'ver{version:06d}'

        # as a fallback assign the last internal folder
        if not os.path.isdir(version_dir):
            version_dir = inner_dirs[-1]

        version_path = os.path.join(base_path, version_dir)
        confidential_model = os.path.join(version_path,'confidential_model.yaml')
        
        # check if it is a confidential model
        if (os.path.isfile(confidential_model)):

            confidential = True

            flame_source = os.path.dirname(os.path.abspath(__file__))
            children_source = os.path.join(flame_source, 'children')
            children_names = ['apply', 'idata', 'odata', 'learn', 'slearn', 'sapply']

            for cname in children_names:
                cpath = os.path.join(children_source, cname+'_child.py')
                shutil.copy(cpath, version_path)
                LOG.info (f'Adding local children: {cpath} ...')

            # open confidential_model.yaml
            with open(confidential_model, 'r') as fc:
                cmodel = yaml.safe_load (fc)

            # create model-results.pkl
            model_building_info =  [('nobj','',cmodel['nobj'])] 
            model_building_info += [('nvarx','',cmodel['nvarx'])] 
            model_building_info += [('model','',cmodel['model'])]

            model_validation_info = [('R2','',cmodel['R2'])] 
            model_validation_info += [('Q2','',cmodel['Q2'])]
            model_validation_info += [('SDEC','',cmodel['SDEC'])]
            model_validation_info += [('SDEP','',cmodel['SDEP'])]
            model_validation_info += [('scoringP','',cmodel['scoringP'])]
            model_validation_info += [('scoringR','',cmodel['scoringR'])]

            model_type_info =  [('quantitative','',cmodel['quantitative'])] 
            model_type_info += [('conformal','',cmodel['conformal'])] 
            model_type_info += [('conformal_confidence','',cmodel['conformal_confidence'])]
            model_type_info += [('ensemble','',False)]  
            model_type_info += [('ensemble_names','',[])]  
            model_type_info += [('ensemble_versions','',[])]  
            model_type_info += [('confidential','',True)]  
            model_type_info += [('secret','',True)]  

            conveyor = Conveyor()
            conveyor.addMeta('modelID', cmodel['modelID'])
            conveyor.addMeta('endpoint', endpoint)
            conveyor.addMeta('version', version)
            conveyor.addMeta('quantitative', True)

            conveyor.addVal (model_building_info,
                    'model_build_info',
                    'model building information',
                    'method',
                    'single',
                    'Information about the model building')
            conveyor.addVal (model_validation_info,
                    'model_valid_info',
                    'model validation information',
                    'method',
                    'single',
                    'Information about the model validation')
            conveyor.addVal (model_type_info,
                    'model_type_info',
                    'model type information',
                    'method',
                    'single',
                    'Information about the model type')

            results_file_name = os.path.join(version_path,'model-results.pkl')
            with open(results_file_name, 'wb') as handle:
                conveyor.save (handle)

        # clone the version in dev
        shutil.copytree(version_path, os.path.join(base_path, 'dev'))
        LOG.info (f'Cloning version {version} to version 0 ...')

    if confidential:
        LOG.info(f'Import of CONFIDENTIAL model {model} version {version} was successfull')
        return True, 'OK'
    
    # get libraries
    message = f'Endpoint {endpoint} imported OK'
    for x in os.listdir(base_path):
        model_path = os.path.join(base_path,x)
        model_pkl  = os.path.join(model_path,'estimator.pkl')
        dict_estimator = {}
        if os.path.isfile (model_pkl):
            with open(model_pkl, "rb") as input_file:
                try:
                    dict_estimator = pickle.load(input_file)
                except Exception as e:
                    return False, f'Incompatible libraries found!. Import aborted with message "{str(e)}"'

        # check if the libraries used to build this model are similar to current libraries
        if 'libraries' in dict_estimator:
            # print (dict_estimator['libraries'])
            success, results = utils.compatible_modules(dict_estimator['libraries'])
            if not success:
                message = f"WARNING: Incompatible libraries detected, {results}. Use at your own risk"
                return False, message

    LOG.info('Libraries used to generate the imported model are compatible with local libraries')
    LOG.info(message)
    return True, message


def action_export(model, version=0):
    '''
    Exports the whole model tree indicated in the argument as a single
    tarball file with the same name.
    '''

    confidential_files = ['confidential_model.yaml', 'confidential_preprocess.yaml', 
                          'parameters.yaml', 'documentation.yaml']
    if not model:
        return False, 'Empty model label'

    current_path = os.getcwd()

    base_path = utils.model_tree_path(model)
    if not os.path.isdir(base_path):
        return False, 'Unable to export, endpoint directory not found'

    # change to model repository to tar the file from there
    os.chdir(base_path)

    confidential = False

    # single version export
    if version != 0:
        compressedname = f'{model}_v{version:06d}.tgz'
        compressedfile = os.path.join(current_path,compressedname)
        iversion = f'ver{version:06d}'

        if (os.path.isfile(os.path.join(iversion,'confidential_model.yaml'))):
            confidential = True
            with tarfile.open(compressedfile, 'w:gz') as tar:
                for ifile in confidential_files:
                    ipath = os.path.join (iversion, ifile)
                    if os.path.isfile (ipath):
                        tar.add(ipath)
        else:
            with tarfile.open(compressedfile, 'w:gz') as tar:
                tar.add(iversion)
    else:
        compressedname = model+'.tgz'
        compressedfile = os.path.join(current_path,compressedname)
        itemend = os.listdir()
        itemend.sort()

        # t1 = time.time()

        with tarfile.open(compressedfile, 'w:gz') as tar:
            for iversion in itemend:
                if not os.path.isdir(iversion):
                    continue
                tar.add(iversion)

    # print (time.time()-t1)

    # touch a file o indicate that the compressed file is ready
    file = open(os.path.join(current_path,model+'.completed'),'w')
    file.close()

    # return to current directory
    os.chdir(current_path)

    message = ''
    if confidential:
        message = f'Model {model} version {version} was exported in CONFIDENTIAL mode as {compressedname}'
    else:
        if version !=0 :
            message = f'Model {model} version {version} was exported as {compressedname}'
        else:
            message = f'All the versions for model {model} were exported as {compressedname}'

    LOG.info(message)
    return True, message


def action_series(model, version):
    '''
    Returns the training series used for building a certain model /version 
    '''
    if model is None:
        return False, 'Empty model label'

    meta_path = utils.model_path(model, version)
    training_file = os.path.join(meta_path, 'training_series')
    if not os.path.isfile(training_file):
        return False, 'training series file not found'

    shutil.copy(training_file, './training_series.sdf')
    
    LOG.info(f'Training series for model {model}, version {version}, saved as "training_series.sdf"')
    return True, 'OK'


def action_info(model, version, output='text'):
    '''
    Returns a text or an object with results info for a given model and version
    TODO: add Q/C + conf/no-conf + ensem/no-ensem + list of ensemble (when applicable)
    '''

    if model is None:
        if output != 'text':
            return False, {'code':1, 'message': 'Empty model label'}
        return False, 'Empty model label'


    meta_path = utils.model_path(model, version)
    # Check that both the meta and the results file are present
    # return error code 0 if not
    meta_file = os.path.join(meta_path, 'model-meta.pkl')
    if not os.path.isfile(meta_file):

        # this file is created uppon task abortion
        error_file = os.path.join(tempfile.gettempdir(),'building_'+model)
        if os.path.isfile(error_file):
            with open(error_file, 'r') as f:
                error_text = f.read()
            f.close()
            os.remove(error_file)
            if output != 'text':
                return False, {'code':1, 'message': error_text}
            return False, 'model building task aborted'  
        
        conf_file = os.path.join (meta_path, 'confidential_model.yaml' )
        if os.path.isfile(conf_file):

            confidential_info = ['confidential', 'secret', 'nobj', 'nvarx', 
            'conformal', 'conformal_confidence', 'modelID', 'quantitative', 'model']      
            try:      
                with open(conf_file, 'r') as fc:
                    cmodel = yaml.safe_load (fc)
            except:
                return False, 'Not found'
            info = [('confidential','unique model ID', True)]

            for ielement in confidential_info:
                if ielement in cmodel:
                    info+=[(ielement, 'na', cmodel[ielement] )]

            return True, info

        # return error
        if output != 'text':
            return False, {'code':0, 'message': 'Info file not found'}
        return False, 'Info file not found'

    results_file = os.path.join(meta_path, 'model-results.pkl')
    if not os.path.isfile(results_file):
        if output != 'text':
            return False, {'code':0, 'message': 'Results file not found'}
        return False, 'Results file not found'    

    with open(meta_file,'rb') as handle:
        modelID = pickle.load(handle)
        errorMessage = pickle.load(handle)
        warningMessage = pickle.load(handle)
        build_info = pickle.load(handle)
        valid_info  = pickle.load(handle)
        type_info = pickle.load(handle)

    if errorMessage is not None:
        if output != 'text':
            return False, {'code':1, 'message': errorMessage}
        return False, errorMessage        

    warning_info = None
    if warningMessage is not None:
        warning_info = [('warning', 'runtime warning', warningMessage)]

    # merge everything 
    info = None

    for iinfo in (build_info, valid_info, type_info, warning_info, [('modelID','unique model ID', modelID)]):
        if info == None:
            info = iinfo
        else:
            if iinfo != None:
                info+=iinfo

    if info == None:
        if output != 'text':
            return False, {'code':1, 'message': 'No relevant information found'}
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

    return True, info

def action_results(model, version=None, ouput_variables=False):
    ''' Returns an object with whole results info for a given model and version '''

    if model is None:
        return False, {'code':1, 'message': 'Empty model label'}

    results_path = utils.model_path(model, version)
    results_file = os.path.join(results_path,'model-results.pkl')

    if not os.path.isfile(results_file):
        return False, {'code':0, 'message': 'Results file not found'}

    from flame.conveyor import Conveyor
    conveyor = Conveyor()

    with open(results_file, 'rb') as handle:
        conveyor.load(handle)

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

    if oformat != 'text':
        return True, param

    else:
        yaml = param.dumpYAML()
        for line in yaml:
            print (line)

        return True, 'parameters listed'


## the following commands are argument-less, intended to be called from a web-service to 
## generate python output only

def action_documentation(model, version=None, doc_file=None, oformat='YAML'):
    ''' Returns an object with whole results info for a given model and version '''

    if model is None:
        return False, 'Empty model label'

    from flame.documentation import Documentation
    
    # get de model repo path
    rdir = utils.model_path(model, version)

    if not os.path.isfile(os.path.join(rdir, 'model-results.pkl')):
        return False, 'Info file not found' 

    doc = Documentation(model, version)

    # INPUT, the doc_file will be applied to doc as a delta
    if doc_file is not None:
        success, message = doc.delta(model, version, doc_file, iformat=oformat)
        return success, message

    # CONSOLE PRINTING
    if oformat == 'YAML':
        yaml = doc.dumpYAML ()
        for line in yaml:
            line = line.encode("ascii", "ignore")
            line = line.decode("ascii", "ignore")
            print (line)
        return True, 'parameters listed'
    
    if oformat == 'WORD':
        oname = 'documentation_'+model+'v'+str(version)+'.docx'

        success, result = doc.dumpWORD (oname)

        if success:
            print (f'model documentation saved as: {oname}')
            return True, oname
    
    if oformat == "EXCEL":
        oname = 'documentation_'+model+'v'+str(version)+'.xlsx'

        success, result = doc.dumpExcel(oname)

        if success:
            print(f'model documentation saved as {oname}')
            return True,oname 

        return False, result
    
    # OUTPUT, the doc will be returned as a documentation object
    return True, doc


def action_label(model, version=None, labels=None, oformat='text'):
    ''' Returns / sets the model labels '''

    if model is None:
        return False, 'Empty model label'

    # get de model repo path
    rdir = utils.model_path(model, version)

    if not os.path.isdir(rdir):
        return False, f'Model {model} not found'

    if labels is not None:
        if oformat == 'JSONS':
            try:
                p = json.loads(labels)
            except Exception as e:
                return False, str(e)
        else:
            try:
                with open(labels, 'r') as fi:
                    p = yaml.safe_load(fi)
            except Exception as e:
                return False, e

        for ikey in p:
            if len(p[ikey]) > 15:
                return False, f'labels should be shorter than 15 chars. Label "{ikey} : {p[ikey]}" is {len(p[ikey])} chars long'
            elif len(p[ikey]) < 2:
                return False, f'labels should be longer than 2 chars. Label "{ikey} : {p[ikey]}" is {len(p[ikey])} chars long'
        try:
            with open(os.path.join(rdir, 'model-labels.pkl'), 'wb') as fo:
                pickle.dump(p, fo)
        except Exception as e:
            return False, e

    # open labels 
    try:
        with open(os.path.join(rdir, 'model-labels.pkl'), 'rb') as fi:
            p = pickle.load(fi)
    except Exception as e:
        return False, e

    if oformat == 'text':
        for ikey in p:
            print(f'{ikey}\t{p[ikey]}')
        return True, 'success'

    return True, p

def action_dir():
    '''
    Returns a list of models and versions
    '''
    # get de model repo path
    models_path = pathlib.Path(utils.model_repository_path())
    if models_path.is_dir() is False:
        return False, 'the model repository path does not exist. Please run "flame -c config".'

    # get directories in model repo path
    dirs = [x for x in models_path.iterdir() if x.is_dir()]

    # if dir contains dev/ -> is model (NAIVE APPROACH)
    # get last dir name [-1]: model name
    model_dirs = [d.parts[-1] for d in dirs if list(d.glob('dev'))]

    label_defaults = {  'maturity' : 'dev',
                        'type' : 'unk',
                        'subtype' : 'unk',
                        'endpoint' : 'unk',
                        'species' : 'unk' }
    results = []
    for imodel in model_dirs:
        idict = {}
        idict ["modelname"] = imodel
        idict ["version"] = 0
        idict ["info"] = action_info(imodel, 0, output=None)[1]
        success, label = action_label(imodel, 0, oformat='object')
        if success:
            idict ["label"] = label
        else:
            idict ["label"] = label_defaults

        results.append(idict)

        for iversion in os.listdir(utils.model_tree_path(imodel)):
            if iversion.startswith('ver'):
                idict = {}
                idict ["modelname"] = imodel
                idict ["version"] = utils.modeldir2ver(iversion)
                idict ["info"] = action_info(imodel, idict ["version"], output=None)[1]
                success, label = action_label(imodel, idict ["version"], oformat='object')
                if success:
                    idict ["label"] = label 
                else:
                    idict ["label"] = label_defaults
                results.append(idict)

    # print (results)
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

            # now we have the model name and version, try to get the info
            try:
                isuccess, iresult=action_info(imodel_name, iver, output='bin')
            except:
                continue

            if not isuccess:
                continue
            
            # build a tuple (version, object) for each version and append 
            imodel_vers_info.append((iver, iresult ))

        # build a tuple (model_name, [version_info]) for each model and append
        results.append((imodel_name, imodel_vers_info))
        
    return True, results

def getdate (element):
    return element[0]

def action_predictions_list ():
    '''
    shows a table with the list of predictions 
    '''
    # get de model repo path
    predictions_path = pathlib.Path(utils.predictions_repository_path())
    if predictions_path.is_dir() is False:
        return False, 'the predictions repository path does not exist. Please run "flame -c config".'

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
            try:
                modelID  = pickle.load (handle)
            except:
                modelID = '*legacy*'

        # ifile is simplified to avoid discossing the repository
        ifile = os.path.basename(ifile)

        # ensemble models are hidden
        if label[0:8]=='ensemble':
            continue

        # add as a tupla 
        iresult.append( ( label, endpoint, version, time, ifile, modelID) )

        # format as a text line for reverse date sorting and printing
        line = f'{label:10} {endpoint:15}   {version}   {time}   {ifile} {modelID}'
        result.append( (timestamp, line) )

    result.sort (reverse=True, key = getdate)

    [print (i[1]) for i in result]

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

def action_predictions_result (label, output='text'):
    '''
    try to retrieve the prediction result with the label used as argument
    returns 
        - (False, Null) if it there is no directory or the predictions 
          pickle files cannot be found 
        
        - (True, object) with the results otherwyse
    '''

    # this file is created uppon task abortion
    error_file = os.path.join(tempfile.gettempdir(),'predicting_'+label)
    if os.path.isfile(error_file):
        with open(error_file, 'r') as f:
            error_text = f.read()
        f.close()
        os.remove(error_file)

        if output != 'text':
            return False, {'code':1, 'message': error_text}
        return False, 'model prediction task aborted' 

    # get de model repo path
    predictions_path = pathlib.Path(utils.predictions_repository_path())
    label_path = predictions_path.joinpath(label)
    if not os.path.isdir(label_path):

        if output != 'text':
            return False, {'code':0, 'message': f'directory {label_path} not found'}
        print (f'directory {label_path} not found')
        return False, None

    result_path = label_path.joinpath('prediction-results.pkl')
    if not result_path.is_file():

        if output != 'text':
            return False, {'code':0, 'message': f'predictions not found for {label} directory'}
        print (f'predictions not found for {label} directory')
        return False, None

    iconveyor = Conveyor()

    with open(result_path, 'rb') as handle:
        success, message = iconveyor.load(handle)

    if not success:
        if output != 'text':
            return False, {'code':1, 'message': f'error reading prediction results with message {message}'}
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

    if not os.path.isdir(label_path):
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


def action_refresh (model=None, version=None, GUI=False):
    '''
    Rebuild one or many models making use of existing parameter files and
    locally stored training series. 
    '''

    import flame.context as context
    from flame.parameters import Parameters
    # from flame.documentation import Documentation
    import logging

    if GUI:
        token_file = os.path.join(tempfile.gettempdir(),'refreshing_'+model)
        # update token file with content 'working'    
        with open(token_file, 'w') as f:
            f.write('Analyzing and sorting models...')

    # list endpoints relevant for the arguments
    if model is not None:
        model_list = [model]
    else:
        model_root = pathlib.Path(utils.model_repository_path())
        model_list = [x.stem for x in model_root.iterdir() if x.is_dir()]

    # list versions relevant for the arguments
    task_list = []
    for imodel in model_list:

        if version is not None:
            task_list.append( (imodel, version) )
        else:
            model_root = pathlib.Path(utils.model_tree_path(imodel))
            itask_list = [( imodel, utils.modeldir2ver(x.stem) ) for x in model_root.iterdir() if x.is_dir()]
            task_list+=itask_list  # use "+=" and not "append" to merge the new list with the old one

    # analize task_list and add at the end ensemble models
    # this is needed to have low models refreshed BEFORE refreshing the high models
    # eliminating the need to refresh them recursively 
    LOG.info ("Analyzing and sorting models...")

    # make sure the lower models are in task_list and, if not, force the inclussion
    for itask in task_list:
        param = Parameters()
        success, results = param.loadYaml(itask[0], itask[1])

        if not success:
            continue

        if param.getVal('input_type') == 'model_ensemble':
            ens_nams = param.getVal('ensemble_names')
            ens_vers = param.getVal('ensemble_versions')
            for i in range(len(ens_nams)):
                iver = 0
                inam = ens_nams[i]
                if (i<len(ens_vers)):
                    iver = ens_vers[i]
                if ( (inam,iver) ) not in task_list:
                    task_list.append( (inam, iver) ) 
    
    # create separate lists for regular and ensemble models
    # and add ensemble models at the end
    # this needs to be carried out after the previos step because
    # some of the lower level models could be an ensemble model
    # itself 
    mol_list = []
    ens_list = []
    for itask in task_list:
        param = Parameters()
        success, results = param.loadYaml(itask[0], itask[1])

        if not success:
            mol_list.append(itask)
            continue

        if param.getVal('input_type') == 'model_ensemble':
            ens_list.append(itask)
        else:
            mol_list.append(itask)

    task_list = mol_list + ens_list

    # show all models before stating
    LOG.info ("Starting model refreshing task for the following models and versions")
    for itask in task_list:
        LOG.info (f'   model: {itask[0]}   version: {itask[1]}')
    
    LOG.info ("This can take some time, please be patient...")

    source_dir = os.path.dirname(os.path.abspath(__file__)) 
    children_dir = os.path.join (source_dir, 'children')
    master_parameters = os.path.join (children_dir, 'parameters.yaml') 
    master_documentation = os.path.join (children_dir, 'documentation.yaml') 

    # now send the build command for each task
    for itask in task_list:

        destinat_path = utils.model_path(itask[0],0)        # dev

        if itask[1] != 0:
            # move version to /dev for building
            original_path = utils.model_path(itask[0],itask[1]) # veri  
            security_path = destinat_path+'_security'           # dev_sec
            shutil.move (destinat_path, security_path)          # dev --> dev_sec
            shutil.move (original_path, destinat_path)          # veri --> dev

        LOG.info (f'   refreshing model: {itask[0]}   version: {itask[1]} ({task_list.index(itask)+1} of {len(task_list)})...')
        if GUI:
            with open(token_file, 'w') as f:
                f.write(f'model: {itask[0]} version: {itask[1]} ({task_list.index(itask)+1} of {len(task_list)})')

        # dissable LOG output
        logging.disable(logging.ERROR)

        # update parameters
        dump_parameters = os.path.join (destinat_path, 'parameters_dump.yaml')
        success, param = action_parameters(itask[0], 0, oformat='bin')
        if success:
            param_yaml = param.dumpYAML()
            with open (dump_parameters,'w') as f:
                for line in param_yaml:
                    f.write (line+'\n')
        else:
            LOG.info ('   ERROR: unable to merge parameters for model: {itask[0]}   version: {itask[1]}')
            dump_parameters = None

        original_parameters = os.path.join (destinat_path, 'parameters.yaml')       
        shutil.copy (master_parameters, original_parameters)

        #update documentation
        dump_documentation = os.path.join (destinat_path, 'documentation_dump.yaml')
        success, documentation = action_documentation(itask[0], 0, doc_file= None, oformat='bin')

        original_documentation = os.path.join (destinat_path, 'documentation.yaml')       
        shutil.copy (master_documentation, original_documentation)

        if success:
            documentation_yaml = documentation.dumpYAML()
            with open (dump_documentation,'w') as f:
                for line in documentation_yaml:
                    line = line.encode("ascii", "ignore")
                    line = line.decode("ascii", "ignore")
                    f.write (line+'\n')
            s2, documentation = action_documentation(itask[0], 0, doc_file= None, oformat='bin')
            s3, r3 = documentation.delta(itask[0], 0, dump_documentation)
        else:
            LOG.info ('   ERROR: unable to merge documentation for model: {itask[0]}   version: {itask[1]}')

        # rebuild the model
        command_build = {'endpoint': itask[0], 
                         'infile': None, 
                         'param_file': dump_parameters,
                         'incremental': False}

        success, results = context.build_cmd(command_build)

        # enable LOG output
        logging.disable(logging.NOTSET)
        
        if itask[1] != 0:
            shutil.move (destinat_path, original_path)          # dev --> veri
            shutil.move (security_path, destinat_path)          # dev_sec --> dev

        if not success:
            LOG.error(results)

    LOG.info ("Model refreshing task finished")

    if GUI:
        # update token file with status 'ready'
        with open(token_file, 'w') as f:
            f.write('ready')

    return True, 'OK'

def action_refresh_test (model=None):

    token_file = os.path.join(tempfile.gettempdir(),'refreshing_'+model)
    
    # check fist the presence of an error token. It it exists, return error  
    error_file = os.path.join(tempfile.gettempdir(),'refreshing_error_'+model)
    if os.path.isfile(error_file):
        with (error_file,'r') as f:
            content = f.read()
        os.remove(error_file)
    
        if os.path.isfile(token_file):
            os.remove(token_file)

        return {'status': 'aborted', 'message': content} 

    # if token file does not exist assume it is too early
    if not os.path.isfile(token_file):
        return {'status': 'working', 'message': 'starting'}

    with open(token_file, 'r') as f:
        content = f.read()
        
    if content != 'ready':
        return {'status': 'working', 'message': content}
    
    os.remove(token_file)
    return {'status': 'ready', 'message': 'OK'}



