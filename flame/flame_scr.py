#! -*- coding: utf-8 -*-

# Description    Flame command
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

import argparse
import pathlib
import os
from flame.util import utils, get_logger, config
from flame import __version__
import flame.context as context

LOG = get_logger(__name__)

# # TEMP: only to allow EBI model to run
# def sensitivity(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     return(tp / (tp+fn))


# # TEMP: only to allow EBI model to run
# def specificity(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     return(tn / (tn+fp))

def main():

    LOG.debug('-------------NEW RUN-------------\n')
    parser = argparse.ArgumentParser(
        description=f'Flame version {__version__}. Use Flame to build and manage predictive models or to predict using them.')

    parser.add_argument('-f', '--infile',
                        help='Input file.',
                        required=False)

    parser.add_argument('-e', '--endpoint',
                        help='Endpoint model name.',
                        required=False)

    parser.add_argument('-s', '--space',
                        help='Chemical space name.',
                        required=False)

    parser.add_argument('-v', '--version',
                        help='Endpoint model version.',
                        required=False)

    parser.add_argument('-a', '--action',
                        help='Manage action.',
                        required=False)

    parser.add_argument('-p', '--parameters',
                        help='File with parameters for the current action.',
                        required=False)

    parser.add_argument('-c', '--command',
                        action='store',
                        choices=['predict', 'search', 'build', 'sbuild', 'manage', 'config'],
                        help='Action type: \'predict\' or \'search\' or \'build\' \'sbuild\' or \'manage\' or \'config\'',
                        required=True)

    # parser.add_argument('-log', '--loglevel',
    #                     help='Logger level of verbosity',)

    parser.add_argument('-d', '--directory',
                        help='Defines the root directory for the models and spaces repositories.',
                        required=False)
    parser.add_argument('-t', '--documentation_file',
                        help='File with manually filled documentation fields.',
                        required=False)

    parser.add_argument('-l', '--label',
                        help='Label for facilitating the identification of the prediction.',
                        required=False )

    parser.add_argument('-inc', '--incremental',
                        help='The input file must be added to the existing training series. Only for "build" command.',
                        action='store_true',
                        required=False )

    args = parser.parse_args()

    # init logger Level and set general config
    # another way around would be create a handler with the level
    # and append it to the global instance of logger

    # if args.loglevel:
    #     numeric_level = getattr(logging, args.loglevel.upper(), None)
    #     if not isinstance(numeric_level, int):
    #         raise ValueError('Invalid log level: {}'.format(args.loglevel))
    #     logging.basicConfig(level=numeric_level)

    if args.infile is not None:
        if not os.path.isfile(args.infile):
            LOG.error(f'Input file {args.infile} not found')
            return 

    # make sure flame has been configured before running any command, unless this command if used to 
    # configure flame
    if args.command != 'config':
        utils.config_test()

    if args.command == 'predict':

        if (args.endpoint is None) or (args.infile is None):
            LOG.error('flame predict : endpoint and input file arguments are compulsory')
            return

        version = utils.intver(args.version)

        if args.label is None:
            label = 'temp'
        else:
            label = args.label

        command_predict = {'endpoint': args.endpoint,
                 'version': version,
                 'label': label,
                 'infile': args.infile}

        LOG.info(f'Starting prediction with model {args.endpoint}'
                 f' version {version} for file {args.infile}, labelled as {label}')

        success, results = context.predict_cmd(command_predict)
        if not success:
            LOG.error(results)

    elif args.command == 'search':

        if (args.space is None) or (args.infile is None) :
            LOG.error ('flame search : space and input file arguments are compulsory')
            return

        version = utils.intver(args.version)
        if args.label is None:
            label = 'temp'
        else:
            label = args.label
        
        command_search = {'space': args.space,
                 'version': version,
                 'infile': args.infile,
                 'runtime_param': args.parameters,
                 'label': label}

        LOG.info(f'Starting search on space {args.space}'
                 f' version {version} for file {args.infile}, labelled as {label}')

        success, results = context.search_cmd(command_search)

        if not success:
            LOG.error(results)

    elif args.command == 'build':

        if (args.endpoint is None):
            LOG.error('flame build : endpoint argument is compulsory')
            return

        command_build = {'endpoint': args.endpoint, 
                         'infile': args.infile, 
                         'param_file': args.parameters,
                         'incremental': args.incremental}

        LOG.info(f'Starting building model {args.endpoint}'
                 f' with file {args.infile} and parameters {args.parameters}')

        success, results = context.build_cmd(command_build)

        if not success:
            LOG.error(results)


    elif args.command == 'sbuild':

        if (args.space is None):
            LOG.error('flame sbuild : space argument is compulsory')
            return

        command_build = {'space': args.space, 'infile': args.infile, 'param_file': args.parameters}

        LOG.info(f'Starting building model {args.space}'
                 f' with file {args.infile} and parameters {args.parameters}')

        success, results = context.sbuild_cmd(command_build)

        if not success:
            LOG.error(results)

    elif args.command == 'manage':
        success, results = context.manage_cmd(args)
        if not success:
            LOG.error(results)

    elif args.command == 'config':
        success, results = config(args.directory, (args.action == 'silent'))
        if not success:
            LOG.error(f'{results}, configuration unchanged')
        

# import multiprocessing

if __name__ == '__main__':
    # used to reproduce speed problems in Linux platforms
    # multiprocessing.set_start_method('spawn')

    main()
