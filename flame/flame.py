#! -*- coding: utf-8 -*-

##    Description    Flame command
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import argparse
import shutil

from predict import Predict
from build import Build
import util.utils as utils 
import manage 

def predict_cmd(args):
    ''' Instantiates a Predict object to run a prediction using the given input file and model '''

    version = utils.intver(args.version)

    predict = Predict(args.infile, args.endpoint, version)
    success, results = predict.run()
    print('flame : ', success, results)

def build_cmd(args):
    ''' Instantiates a Build object to build a model using the given input file (training series) and model (name of endpoint, eg. 'CACO2') '''
    
    build = Build(args.infile, args.endpoint)
    success, results = build.run()
    print('flame : ', success, results)

def manage_cmd(args):
    ''' Instantiates a Build object to build a model using the given input file (training series) and model (name of endpoint, eg. 'CACO2') '''
    
    version = utils.intver(args.version)

    if args.action == 'new':
        success, results = manage.action_new(args.endpoint)
    elif args.action == 'kill':
        success, results = manage.action_kill(args.endpoint)
    elif args.action == 'remove':
        success, results = manage.action_remove(args.endpoint, version)
    elif args.action == 'publish':
        success, results = manage.action_publish(args.endpoint)
    elif args.action == 'list':
        success, results = manage.action_list (args.endpoint)
    elif args.action == 'import':
        success, results = manage.action_import (args.endpoint)
    elif args.action == 'export':
        success, results = manage.action_export (args.endpoint)
    elif args.action == 'refactoring':
        success, results = manage.action_refactoring (args.file)
    elif args.action == 'dir':
        success, results = manage.action_dir (args.endpoint)

    print('flame : ', success, results)

def main():

    parser = argparse.ArgumentParser(description='Use Flame to either build a model from or apply a model to the input file.')
    
    parser.add_argument('-f', '--infile', 
        help='Input file.', 
        required=False)

    parser.add_argument('-e', '--endpoint', 
        help='Endpoint model name.', 
        required=False)

    parser.add_argument('-v', '--version', 
        help='Endpoint model version.', 
        required=False)

    parser.add_argument('-a', '--action', 
        help='Manage action.', 
        required=False)

    parser.add_argument('-c', '--command', 
        action='store', 
        choices=['predict', 'build', 'manage'], 
        help='Action type: \'predict\' or \'build\' or \'manage\'' , 
        required=True)

    args = parser.parse_args()
    
    if args.command == 'predict':
        predict_cmd(args)
    elif args.command == 'build':
        build_cmd(args)
    elif args.command == 'manage':
        manage_cmd(args)

if __name__ == '__main__':    
    main()
