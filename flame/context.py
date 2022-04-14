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
# import pathlib
# import sys
import codecs
import string
import re 

from flame.util import utils, verify, get_logger

# if the number of models is higher, try to run in multithread
MAX_MODELS_SINGLE_CPU = 4

LOG = get_logger(__name__)

def get_ensemble_input(task, model_names, model_versions, infile):
    '''
    Manage obtention of input data from a list of models
    '''

    num_models = len (model_names)
    
    # when there are multiple external sources it is more convenient parallelize the 
    # models than to run internal task in parallel

    parallel = (utils.isSingleThread() == False and num_models > MAX_MODELS_SINGLE_CPU)
    
    # disables internal parallelism
    # if parallel:
    #     task.set_single_CPU() 

    # add input molecule to the model input definition of every internal model
    model_suc = []  # True / False
    model_res = []  # conveyor or every prediction, as produced by odata.run_apply

    model_cmd = []
    for i in range(num_models):
        model_cmd.append({'endpoint': model_names[i],
                          'version': model_versions[i],
                          'infile': infile,
                          'output_format': 'ghost',
                          'label': f'ensemble{i}'})

    # run in multithreading
    if parallel:
        # import multiprocessing as mp

        LOG.info(f'Runing {num_models} threads in parallel')       
       
        # pool = mp.Pool(len(model_cmd))
        # model_tmp = pool.map(predict_cmd, model_cmd)

        from joblib import Parallel, delayed
        model_tmp = Parallel(n_jobs=num_models)(delayed(predict_cmd)(model_cmd[i]) for i in range(num_models))

        for iresult in model_tmp:
            model_suc.append(iresult[0])
            model_res.append(iresult[1])
    
    # run in a single thread
    else:
        for i in range(num_models):
            success, results = predict_cmd(model_cmd[i])
            model_suc.append(success)
            model_res.append(results)

    if False in model_suc:
        return False, 'Some external input sources failed: '+str(model_suc)

    LOG.info('External input computed')

    return True, model_res

# def safe_copy (inputfile, outputfile):
#     ''' this function makes sure that the input file contains only printable chars
#         RDKit is very sensitive to the presence of non utf-8 chars and for this reason
#         this pre-filter is needed
#     '''

#     characters_to_keep = string.printable #printable us-ascii only
#     search_regex = re.compile("[^%s]" % (re.escape(characters_to_keep)))

#     read_stream  = codecs.open(inputfile ,'r',encoding='utf-8', errors='ignore') 
#     write_stream = codecs.open(outputfile,'w',encoding='utf-8', errors='ignore')
 
#     buffer = 'start'                                                        
#     buffer_size = 512*1024 # size in bytes. -1 for loading whole file in 

#     while  buffer: # empty string evaluates as False. Any other string as True.
#         buffer = read_stream.read(buffer_size)
#         write_stream.write(search_regex.sub('?', buffer))

#     read_stream.close()
#     write_stream.close()


def predict_cmd(arguments, output_format=None):
    '''
    Instantiates a Predict object to run a prediction using the given input
    file and model.

    This method must be self-contained and suitable for being called in
    cascade, by models which use the output of other models as input.
    '''
    from flame.predict import Predict

    # safety check if model exists
    endpoint_dir = utils.model_path(arguments['endpoint'], 0)
    if not os.path.isdir(endpoint_dir):
        return False, 'Endpoint name not found in model repository.'

    # ** DEPRECATE **
    # this is a back-compatibility trick for older versions of APIs 
    # not supporting the label argument
    if 'label' not in arguments:
        arguments['label'] = 'temp'

    if 'output_format' in arguments:
        output_format = arguments['output_format']

    predict = Predict(arguments['endpoint'], version=arguments['version'],  output_format=output_format, label=arguments['label'])

    if utils.isSingleThread():
        predict.set_single_CPU()

    ensemble = predict.get_ensemble()

    # ensemble[0]     Boolean with True for ensemble models and False otherwyse
    # ensemble[1]     List of ensemble model model_names
    # ensemble[2]     List of ensemble model versions

    if ensemble[0]:

        if arguments['infile'] is None:
            return False, 'ensemble models require allways an input file'

        emodels = ensemble[1]
        evers   = ensemble[2]

        success, model_res = get_ensemble_input(predict, emodels, evers, arguments['infile'])

        if not success:
            predict.conveyor.setError (model_res)
            LOG.error (model_res)
            # return False, model_res        # TO-DO, comment this line and run prediction to allow odata to generate error info

        # check the presence of changes in the inner models
        modelID = predict.conveyor.getMeta('modelID')
        for i in range(len (emodels)):
            success, iID = utils.getModelID(emodels[i], evers[i], 'model')
            if success:
                if iID not in modelID:
                    predict.conveyor.setWarning (f'Inner model {emodels[i]}.{evers[i]} has been updated. Rebuilding of ensemble model is recommended')
                    LOG.warning (f'Inner model {emodels[i]}.{evers[i]} has been updated. Rebuilding of ensemble model is recommended')

        # now run the model using the data from the external sources
        success, results = predict.run(model_res)

    else:

        # run the model with the input file
        # utils.safe_copy(arguments['infile'], './tempsafe')
        # success, results = predict.run('./tempsafe')

        success, results = predict.run(arguments['infile'])

    LOG.info('Prediction completed...')

    return success, results

def profile_cmd (arguments, output_format=None):
    '''
    Instantiates a Predict object to run a prediction using the given input
    file and model.

    This method must be self-contained and suitable for being called in
    cascade, by models which use the output of other models as input.
    '''
    from flame.predict import Predict

    if 'label' not in arguments:
        arguments['label'] = 'temp'

    if 'output_format' in arguments:
        output_format = arguments['output_format']

    predict = Predict('multi', 0,  output_format=output_format, label=arguments['label'],profile=True)

    if utils.isSingleThread():
        predict.set_single_CPU()
    
    predict.param.setVal('input_type', 'molecule')
    predict.param.setVal('SDFile_name', ['name', 'GENERIC_NAME'])
    predict.param.setVal('output_format', ['JSON'])

    if arguments['infile'] is None:
        return False, 'multi models require allways an input file'

    emodels = arguments['multi']['endpoints']
    evers   = arguments['multi']['versions']

    success, model_results = get_ensemble_input(predict, emodels, evers, arguments['infile'])

    if not success:
        predict.conveyor.setError (model_results)
        LOG.error (model_results)

    success, results =  predict.aggregate(model_results, arguments['infile'])

    LOG.info('Profiling completed...')

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
    endpoint_dir = utils.model_path(arguments['endpoint'], 0)
    if not os.path.isdir(endpoint_dir):
        return False, 'Endpoint name not found in model repository.'

    # remove pre-existing results file
    results_file = os.path.join(endpoint_dir, 'model-results.pkl')
    if os.path.isfile(results_file):
        os.remove(results_file)

    meta_file = os.path.join(endpoint_dir, 'model-meta.pkl')
    if os.path.isfile(meta_file):
        os.remove(meta_file)

    # input file provided in the command
    ifile = arguments['infile']
    if ifile is not None and not os.path.isfile (ifile):
        return False, f'Wrong training series file {ifile}'

    # lfile is the "training_series" copied internally to the endpoint folder
    endpoint_path = utils.model_path(arguments['endpoint'], 0)
    lfile = os.path.join(endpoint_path, 'training_series')

    if 'param_file' in arguments:
        build = Build(arguments['endpoint'], param_file=arguments['param_file'], output_format=output_format)
    elif 'param_string' in arguments:
        build = Build(arguments['endpoint'], param_string=arguments['param_string'], output_format=output_format)
    else:
        build = Build(arguments['endpoint'], output_format=output_format)

    if utils.isSingleThread():
        build.set_single_CPU()

    ensemble = build.get_ensemble()

    # ensemble[0]     Boolean with True for ensemble models and False otherwyse
    # ensemble[1]     List of ensemble model model_names
    # ensemble[2]     List of ensemble model versions

    if ensemble[0]:

        emodels = ensemble[1]
        evers   = ensemble[2]

        if ifile is None:
            if not os.path.isfile (lfile):
                return False, 'no training series detected'
        else:
            try:
                utils.safe_copy(ifile, lfile)
                # shutil.copy(ifile, lfile)
            except:
                return False, 'Unable to copy input file to model directory'
        
        success, model_res = get_ensemble_input(build, emodels, evers, lfile)

        if not success:
            return False, model_res

        for i in range(len (emodels)):
            success, iID = utils.getModelID(emodels[i], evers[i], 'model')
            if success:
                build.extend_modelID(iID)

        LOG.debug(f'New modelID is: {build.conveyor.getMeta("modelID")}')

        # now run the model using the data from the external sources
        success, results = build.run(model_res)

    else:

        # when a new training series is provided in the command line
        # try to copy it to the model directory
        if ifile is not None:

            # in case of incremental training, add the input file at the end of existing file
            if arguments['incremental']:
                if arguments['incremental'] and os.path.isfile(lfile):
                    LOG.info(f'Merging file {ifile} with existing training series')
                    new_training = os.path.join(endpoint_path, 'temp_training')

                    characters_to_keep = string.printable #printable us-ascii only
                    search_regex = re.compile("[^%s]" % (re.escape(characters_to_keep)))

                    with open(new_training, 'w') as outfile:

                        # handling the extra newline of SDFiles is problematic. We are delaying the
                        # output of the newline by striping newlines and adding an universal newline
                        # at the next line for the first block  
                        first = True
                        with codecs.open(lfile, 'r', encoding='utf-8', errors='ignore') as infile:
                            for line in infile:
                                line = search_regex.sub('?', line)
                                if first:
                                    outfile.write(f'{line.rstrip()}')
                                    first = False
                                else:
                                    outfile.write(f'\n{line.rstrip()}')

                        # for the second block we add the preceding newline in all lines 
                        with codecs.open(ifile, 'r', encoding='utf-8', errors='ignore') as infile:
                            for line in infile:
                                line = search_regex.sub('?', line)
                                outfile.write(f'\n{line.rstrip()}')

                    shutil.move(new_training, lfile)
            else:
                try:
                    utils.safe_copy (ifile, lfile)
                    # shutil.copy(ifile, lfile)
                except:
                    return False, 'Unable to copy input file to model directory'

        # check that the local copy of the input file exists
        # if not os.path.isfile(lfile):
        #     return False, 'No training series found'

        # run the model with the input file
        success, results = build.run(lfile)

    return success, results

def sbuild_cmd(arguments, output_format=None):
    '''
    Instantiates a Sbuild object to build a chemical space using the given
    input file and model. 

    '''
    
    from flame.sbuild import Sbuild

    # safety check if model exists
    space_dir = utils.space_path(arguments['space'], 0)
    if not os.path.isdir(space_dir):
        return False, 'Endpoint name not found in space repository.'

    # remove pre-existing results file
    results_file = os.path.join(space_dir, 'space-results.pkl')
    if os.path.isfile(results_file):
        os.remove(results_file)
    meta_file = os.path.join(space_dir, 'space-meta.pkl')
    if os.path.isfile(meta_file):
        os.remove(meta_file)


    if 'param_string' in arguments:
        sbuild = Sbuild(arguments['space'], param_string=arguments['param_string'], output_format=output_format)
    elif 'param_file' in arguments:
        sbuild = Sbuild(arguments['space'], param_file=arguments['param_file'], output_format=output_format)
    else:
        sbuild = Sbuild(arguments['space'], output_format=output_format)

    if utils.isSingleThread():
        sbuild.set_single_CPU()

    ifile = arguments['infile']
    epd = utils.space_path(arguments['space'], 0)
    lfile = os.path.join(epd, 'training_series')

    # when a new training series is provided in the command line
    # try to copy it to the model directory
    if ifile is not None:
        if not os.path.isfile(ifile):
            return False, f'Wrong compound database file {ifile}'
        try:
            utils.safe_copy(ifile, lfile)
            # shutil.copy(ifile, lfile)
        except:
            return False, 'Unable to copy input file to space directory'

    # check that the local copy of the input file exists
    if not os.path.isfile(lfile):
        return False, 'No compound database found'

    # run the space building with the input file
    success, results = sbuild.run(lfile)

    return success, results

def search_cmd(command, output_format=None):
    '''
    Instantiates a Search object to run a search using the given input
    file and space.

    '''
    from flame.search import Search

    # ** DEPRECATE **
    # this is a back-compatibility trick for older versions of APIs 
    # not supporting the label argument

    if 'label' not in command:
        command['label'] = 'temp'

    # safety check if model exists
    space_dir = utils.space_path(command['space'], 0)
    if not os.path.isdir(space_dir):
        return False, 'Endpoint name not found in space repository.'

    search = Search(command['space'], version=command['version'], output_format=output_format, label=command['label'])

    if utils.isSingleThread():
        search.set_single_CPU()

    success, results = search.run(command)

    LOG.info('Search completed...')

    return success, results

def manage_cmd(args):
    '''
    Calls diverse model or space maintenance commands
    '''

    version = utils.intver(args.version)

    model_item = 0
    if args.item is not None:
        try:
            model_item = int(args.item)
        except:
            pass

    if args.space is not None or 'searches' in args.action :
    
        import flame.smanage as smanage
    
        if args.action == 'new':
            success, results = smanage.action_new(args.space)
        elif args.action == 'kill':
            success, results = smanage.action_kill(args.space)
        elif args.action == 'remove':
            success, results = smanage.action_remove(args.space, version)
        elif args.action == 'publish':
            success, results = smanage.action_publish(args.space)
        elif args.action == 'list':
            success, results = smanage.action_list(args.space)
        elif args.action == 'parameters':
            success, results = smanage.action_parameters(args.space, version)
        elif args.action == 'info':
            success, results = smanage.action_info(args.space, version)
        elif args.action == 'dir':
            success, results = smanage.action_dir()
        elif args.action == 'searches_result':
            success, results = smanage.action_searches_result(args.label)
        else: 
            success = False
            results = "Specified manage action is not defined"
    else: 

        import flame.manage as manage

        if args.action == 'new':
            success, results = manage.action_new(args.endpoint)
        elif args.action == 'kill':
            success, results = manage.action_kill(args.endpoint)
        elif args.action == 'remove':
            success, results = manage.action_remove(args.endpoint, version)
        elif args.action == 'publish':
            success, results = manage.action_publish(args.endpoint)
        elif args.action == 'list':
            success, results = manage.action_list(args.endpoint)
        elif args.action == 'export':
            success, results = manage.action_export(args.endpoint, version)
        elif args.action == 'info':
            success, results = manage.action_info(args.endpoint, version)
        elif args.action == 'refresh':
            if args.version == None:
                version = None
            success, results = manage.action_refresh(args.endpoint, version)
        elif args.action == 'series':
            success, results = manage.action_series(args.endpoint, version)         
        elif args.action == 'results':
            success, results = manage.action_results(args.endpoint, version)
        elif args.action == 'parameters':
            success, results = manage.action_parameters(args.endpoint, version)
        elif args.action == 'documentation':
            success, results = manage.action_documentation(args.endpoint, version,
            args.documentation_file)
        elif args.action == 'documentation_word':
            success, results = manage.action_documentation(args.endpoint, version, 
            None, oformat='WORD')
        elif args.action == 'documentation_excel':
            success, results = manage.action_documentation(args.endpoint, version, 
            None, oformat='EXCEL')
        elif args.action == 'model_template':
            success, results = manage.action_model_template(args.endpoint, 
            version,  args.documentation_file)
        elif args.action == 'prediction_template':
            success, results = manage.action_prediction_template(args.endpoint, version)
        elif args.action == 'import':
            success, results = manage.action_import(args.infile)
        elif args.action == 'dir':
            success, results = manage.action_dir()
        elif args.action == 'report':
            success, results = manage.action_report()
        elif args.action == 'list':
            success, results = manage.action_list(args.endpoint)
        elif args.action == 'predictions':
            success, results = manage.action_predictions_list()
        elif args.action == 'profiles':
            success, results = manage.action_profiles_list()
        elif args.action == 'predictions_result':
            success, results = manage.action_predictions_result(args.label)
        elif args.action == 'profiles_result':
            success, results = manage.action_profiles_result(args.label, model_item)
        elif args.action == 'profiles_summary':
            success, results = manage.action_profiles_summary(args.label)
        elif args.action == 'predictions_remove':
            success, results = manage.action_predictions_remove(args.label)
        elif args.action == 'profiles_remove':
            success, results = manage.action_profiles_remove(args.label)
        elif args.action == 'label':
            success, results = manage.action_label(args.endpoint, version, args.label)
        elif args.action == 'verify':
            success, results = verify.verify(args.endpoint, version)
        else: 
            success = False
            results = "Specified manage action is not defined"

    return success, results