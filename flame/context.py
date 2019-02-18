#! -*- coding: utf-8 -*-

# Description    Context wrapps calls to predict and build to
# support models making use of extenal input sources
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
import shutil
import pathlib

from flame.util import utils, get_logger

# if the number of models is higher, try to run in multithread
MAX_MODELS_SINGLE_CPU = 4

LOG = get_logger(__name__)

def get_external_input(task, model_set, infile):
    '''
    Manage obtention of input data from external
    data sources (e.g. models or MD servers)
    '''

    # parallel is approppriate for many external sources
    parallel = (len(model_set) > MAX_MODELS_SINGLE_CPU)
    if parallel:
        task.set_single_CPU()

    # add input molecule to the model input definition of every internal model
    for mi in model_set:
        mi['infile'] = infile

    model_suc = []
    model_res = []

    # TODO: if any of the models belongs to another module, send a POST for
    # obtaining the results

    if parallel:

        import multiprocessing as mp
        
        pool = mp.Pool(len(model_set))
        model_temp = pool.map(predict_cmd, model_set)

        for x in model_temp:
            model_suc.append(x[0])
            model_res.append(x[1])
    else:
        for mi in model_set:
            success, results = predict_cmd(mi)
            model_suc.append(success)
            model_res.append(results)

    if False in model_suc:
        return False, 'Some external input sources failed: ', str(model_suc)

    LOG.info('Building completed')

    return True, model_res


def predict_cmd(model, output_format=None):
    '''
    Instantiates a Predict object to run a prediction using the given input
    file and model.

    This method must be self-contained and suitable for being called in
    cascade, by models which use the output of other models as input.
    '''
    from flame.predict import Predict

    predict = Predict(model['endpoint'], model['version'], output_format)

    ext_input, model_set = predict.get_model_set()

    if ext_input:

        success, model_res = get_external_input(
            predict, model_set, model['infile'])

        if not success:
            return False, model_res

        # now run the model using the data from the external sources
        success, results = predict.run(model_res)

    else:

        # run the model with the input file
        success, results = predict.run(model['infile'])

    LOG.info('Prediction completed...')

    return success, results


def build_cmd(arguments, output_format=None):
    '''
    Instantiates a Build object to build a model using the given
    input file and model. 

    This method must be self-contained and suitable for being called in
    cascade, by models which use the output of other models as input
    '''
    
    from flame.build import Build

    # safety check if model exists
    repo_path = pathlib.Path(utils.model_repository_path())
    model_list = os.listdir(repo_path)

    if arguments['endpoint'] not in model_list:
        LOG.error('Endpoint name not found in model repository.')
        return False, 'Endpoint name not found in model repository.'

    build = Build(arguments['endpoint'], arguments['parameters'], output_format)

    ext_input, model_set = build.get_model_set()

    if ext_input:

        success, model_res = get_external_input(
            build, model_set, arguments['infile'])

        if not success:
            return False, model_res

        # now run the model using the data from the external sources
        success, results = build.run(model_res)

    else:

        ifile = arguments['infile']
        epd = utils.model_path(arguments['endpoint'], 0)
        lfile = os.path.join(epd, 'training_series')

        # when a new training series is provided in the command line
        # try to copy it to the model directory
        if ifile is not None:
            if not os.path.isfile(ifile):
                LOG.error(f'Wrong training series file {ifile}')
                return False, f'Wrong training series file {ifile}'
            try:
                shutil.copy(ifile, lfile)
            except:
                LOG.error(f'Unable to copy input file to model directory')
                return False, 'Unable to copy input file to model directory'

        # check that the local copy of the input file exists
        if not os.path.isfile(lfile):
            LOG.error(f'No training series found')
            return False, 'No training series found'

        # run the model with the input file
        success, results = build.run(lfile)

    return success, results

def manage_cmd(args):
    '''
    Calls diverse model maintenance commands
    '''

    import flame.manage as manage

    version = utils.intver(args.version)

    if args.action == 'new':
        # check if config model repo path is correct
        utils.check_repository_path()
        success, results = manage.action_new(args.endpoint)
    elif args.action == 'kill':
        success, results = manage.action_kill(args.endpoint)
    elif args.action == 'remove':
        success, results = manage.action_remove(args.endpoint, version)
    elif args.action == 'publish':
        success, results = manage.action_publish(args.endpoint)
    elif args.action == 'list':
        success, results = manage.action_list(args.endpoint)
    elif args.action == 'import':
        success, results = manage.action_import(args.infile)
    elif args.action == 'export':
        success, results = manage.action_export(args.endpoint)
    elif args.action == 'refactoring':
        success, results = manage.action_refactoring(args.file)
    elif args.action == 'info':
        success, results = manage.action_info(args.endpoint, version)
    elif args.action == 'results':
        success, results = manage.action_results(args.endpoint, version)
    elif args.action == 'parameters':
        success, results = manage.action_parameters(args.endpoint, version)
    elif args.action == 'dir':
        success, results = manage.action_dir()
    elif args.action == 'report':
        success, results = manage.action_report()
    else: 
        success = False
        results = "Specified manage action is not defined"

    return success, results