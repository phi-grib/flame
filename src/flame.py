#! -*- coding: utf-8 -*-

##    Description    Flame predict command
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
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import sys
import getopt
from flpredict import flPredict

def predict_cmd (ifile, imodel):
    
    predict = flPredict (ifile, imodel)
    success, results = predict.run()
    print (success, results)

def usage ():
    print ('usage is flame -c predict -e modelname -f inputfilename.sdf')
    sys.exit(1)

def main ():
    ifile= ''
    imodel = ''
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
                imodel = arg

    if ifile == '' or imodel == '' or command == None: 
        usage()
    
    if command == 'predict':
        predict_cmd (ifile, imodel)
    elif command == 'build':
        print ('not enabled, please wait')
        usage()
    else:
        usage()

if __name__ == '__main__':    
    main()
