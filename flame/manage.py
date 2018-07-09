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

from flame.util import utils
import os
import sys
import shutil
import tarfile
import json
import pickle
import pathlib


def set_model_repository(path=None):
    """
    Set the model repository path.
    This is the dir where flame is going to create and load models
    """
    utils.set_model_repository(path)


def action_new(model):
    '''
    Create a new model tree, using the given name.
    This creates the development version "dev",
    copying inside default child classes
    '''

    if not model:
        return False, 'empty model label'

    ndir = utils.model_tree_path(model)

    # check if there is already a tree for this endpoint
    if os.path.isdir(ndir):
        return False, 'This endpoint already exists'

    os.mkdir(ndir)

    ndir += '/dev'
    os.mkdir(ndir)

    wkd = os.path.dirname(os.path.abspath(__file__))
    children_names = ['apply', 'idata', 'odata', 'learn']
    for cname in children_names:
        shutil.copy(wkd+'/children/'+cname+'_child.py',
                    ndir+'/'+cname+'_child.py')
    shutil.copy(wkd+'/children/parameters.yaml', ndir)

    return True, 'new endpoint '+model+' created'


def action_kill(model):
    '''
    removes the model tree described by the argument
    '''

    if not model:
        return False, 'empty model label'

    ndir = utils.model_tree_path(model)

    if not os.path.isdir(ndir):
        return False, 'model not found'

    shutil.rmtree(ndir, ignore_errors=True)

    return True, 'model '+model+' removed'


def action_publish(model):
    '''
    clone the development "dev" version as a new model version,
     assigning a sequential version number
    '''

    if not model:
        return False, 'empty model label'

    bdir = utils.model_tree_path(model)

    if not os.path.isdir(bdir):
        return False, 'model not found'

    v = None
    v = [int(x[-6:]) for x in os.listdir(bdir) if x.startswith("ver")]

    # try:
    #     v = [int(x[-6:]) for x in os.listdir(bdir) if x.startswith("ver")]
    # except:
    #     pass

    if not v:
        max_version = 0
    else:
        max_version = max(v)

    new_dir = bdir+'/ver%0.6d' % (max_version+1)

    if os.path.isdir(new_dir):
        return False, 'version already exists'

    shutil.copytree(bdir+'/dev', new_dir)

    return True, 'development version published as version '+str(max_version+1)


def action_remove(model, version):
    '''
    Remove the version indicated as argument from the model tree indicated
    as argument
    '''

    if not model:
        return False, 'empty model label'

    if version == 0:
        return False, 'development version cannot be removed'

    rdir = utils.model_path(model, version)
    if not os.path.isdir(rdir):
        return False, 'version not found'

    shutil.rmtree(rdir, ignore_errors=True)

    return True, 'version '+str(version)+' of model '+model+' removed'


def action_list(model):
    '''
    Lists available models (if no argument is provided)
     and model versions (if "model" is provided as argument)
    '''

    # TODO: if no argument is provided, also list all models
    if not model:
        rdir = utils.model_repository_path()
        print(rdir)

        num_models = 0
        for x in os.listdir(rdir):
            num_models += 1
            print(x)

        return True, str(num_models)+' models found in the repository'

    bdir = utils.model_tree_path(model)

    num_versions = 0
    for x in os.listdir(bdir):
        if x.startswith("ver"):

            num_versions += 1
            print(model, ':', x)

    return True, 'model '+model+' has '+str(num_versions)+' published versions'


def action_import(model):
    '''
    Creates a new model tree from a tarbal file with the name "model.tgz"
    '''

    if not model:
        return False, 'empty model label'

    # convert model to endpoint string
    base_model = os.path.basename(model)
    endpoint = os.path.splitext(base_model)[0]
    ext = os.path.splitext(base_model)[1]

    bdir = utils.model_tree_path(endpoint)

    if os.path.isdir(bdir):
        return False, 'endpoint already exists'

    if ext != '.tgz':
        importfile = os.path.abspath(model+'.tgz')
    else:
        importfile = model

    print(importfile)

    if not os.path.isfile(importfile):
        return False, 'importing package '+importfile+' not found'

    try:
        os.mkdir(bdir)
        # os.chdir(bdir)
    except:
        return False, 'error creating directory '+bdir

    with tarfile.open(importfile, 'r:gz') as tar:
        tar.extractall(bdir)

    return True, 'endpoint '+endpoint+' imported OK'


def action_export(model):
    '''
    Exports the whole model tree indicated in the argument as a single
    tarball file with the same name.
    '''

    if not model:
        return False, 'empty model label'

    current_path = os.getcwd()
    exportfile = current_path+'/'+model+'.tgz'

    bdir = utils.model_tree_path(model)

    if not os.path.isdir(bdir):
        return False, 'endpoint directory not found'

    os.chdir(bdir)

    itemend = os.listdir()
    itemend.sort()

    with tarfile.open(exportfile, 'w:gz') as tar:
        for iversion in itemend:
            if not os.path.isdir(iversion):
                continue
            tar.add(iversion)

    os.chdir(current_path)

    return True, 'endpoint '+model+' exported as '+model+'.tgz'


# TODO: implement refactoring, starting with simple methods
def action_refactoring(file):
    '''
    NOT IMPLEMENTED,
    call to import externally generated models (eg. in KNIME or R)
    '''

    print('refactoring')

    return True, 'OK'


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

    # print(json.dumps(results))


def action_info(model, version=None, output='text'):
    '''
    Returns a text or JSON with info for a given model and version
    '''

    if model is None:
        return False, 'empty model label'

    if version == None:
        return False, 'no version provided'

    rdir = utils.model_path(model, version)
    if not os.path.isfile(os.path.join(rdir, 'info.pkl')):
        return False, 'info not found'

    with open(os.path.join(rdir, 'info.pkl'), 'rb') as handle:
        results = pickle.load(handle)
        results += pickle.load(handle)

    if output == 'text':
        for val in results:
            if len(val) < 3:
                print(val)
            else:
                print(val[0], ' (', val[1], ') : ', val[2])
        return True, 'model informed OK'

    new_results = []

    # results must be checked to avoid numpy elements not JSON serializable
    for i in results:
        if 'numpy.int64' in str(type(i[2])):
            try:
                v = int(i[2])
            except:
                v = None
            new_results.append((i[0], i[1], v))
        elif 'numpy.float64' in str(type(i[2])):
            try:
                v = float(i[2])
            except:
                v = None
            new_results.append((i[0], i[1], v))
        else:
            new_results.append(i)

    return True, json.dumps(new_results)
