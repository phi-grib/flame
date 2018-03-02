#! -*- coding: utf-8 -*-

##    Description    Flame predict command
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import sys
import argparse
from flpredict import flPredict

def predict (args):
    predict = flPredict (args.infile, args.model)
    success, results = predict.run()
    print (success, results)

def usage ():
    print ('usage is predict -e modelname -f inputfilename.sdf')
    sys.exit(1)

def main ():

    parser = argparse.ArgumentParser(description='Flame command line script to apply a model to an input file.')
    parser.add_argument('-f', '--infile', help='Input file.',
                        default= 'test.sdf', required=True)
    parser.add_argument('-e', '--model', help='Model file.', required=True)
    args = parser.parse_args()
    
    predict (args)    

if __name__ == '__main__':    
    main()