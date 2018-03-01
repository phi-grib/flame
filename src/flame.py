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

import sys
import getopt
from predict import Predict
from build import Build

def predict_cmd(ifile, model):
    ''' Instantiates a Predict object to run a prediction using the given input file and model '''
    
    predict = Predict (ifile, model)
    success, results = predict.run()
    print (success, results)

def build_cmd(ifile, model):
    ''' Instantiates a Build object to run a prediction using the given input file and model '''
    
    Build = Build (ifile, model)
    success, results = build.run()
    print (success, results)

def usage ():
    ''' Usage instructions '''
    
    print ('usage is flame -c predict -e modelname -f inputfilename.sdf')
    sys.exit(1)

def main ():
    ifile= ''
    model = ''
    command = None

    try:
       opts, args = getopt.getopt(sys.argv[1:],'c:f:e:')
    except getopt.GetoptError:
        print("Arguments not recognized")
        usage()
    
    if not len(opts):
        print("Arguments not recognized")
        usage()

    if len(opts)>0:
        for opt, arg in opts:

            if opt in '-c':
                command = arg
            elif opt in '-f':
                ifile = arg
            elif opt in '-e':
                model = arg

    if ifile == '' or model == '' or command == None: 
        usage()
    
    if command == 'predict':
        predict_cmd(ifile, model)
    elif command == 'build':
        build_cmd(ifile, model)
    else:
        usage()

if __name__ == '__main__':    
    main()
