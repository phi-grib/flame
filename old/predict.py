#!/usr/bin/env python

# -*- coding: utf-8 -*-

##    Description    eTOXlab component for runing a predictive model
##                   
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu) 
##
##    Copyright 2013 Manuel Pastor
##
##    This file is part of eTOXlab.
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
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import getopt
import cPickle as pickle

from utils import lastVersion
from utils import exposedVersion
from utils import writeError
from utils import removefile

def predict (endpoint, molecules, verID=-1, api=0, loc=-99, detail=False, progress=False, extValid=False):
    """Top level prediction function

       molecules:  SDFile containing the collection of 2D structures to be predicted
       verID:      version of the model that will be used. Value -1 means the last one
       detail:     level of detail of the prediction. If True the structure of the
                   closest compond will be returned
    """

    # web calls, we look for web exposed versions
    if api in (1,2,5):
        vpath = exposedVersion (endpoint)
        if vpath == None:
            return (False, 'no published model found')
    else:
        vpath = lastVersion (endpoint,verID)  ## also for api==6 (API v3.0)
        
    if not vpath:
        return (False,"No versions directory found")

    ##print vpath
    
    if loc != -99:
        vpath += '/local%0.4d' % loc
    
    sys.path.append(vpath)
    
    from imodel import imodel

    # load model
    model = imodel(vpath)

    # to maintain back-compatibility, the last parameter is only introduced when set to TRUE
    # this is used only in command mode (-x flag) and therefore never set for web-based preditions
    if extValid:
        success, pred = model.predictWorkflow (molecules, detail, progress, extValid)
    else:
        success, pred = model.predictWorkflow (molecules, detail, progress)

    return (success, pred)

def presentPredictionText (pred):
    
    """Writes the result of the prediction into a log file and prints some of them in the screen
    """
    
    if pred[0]:
        for x in pred[1]:
            if x[0]:
                for y in x[1]:
                    if y[0]:
                        if isinstance(y[1], float):
                            print "%8.5f" % y[1],
                        else:
                            print y[1],
                    else:
                        print y,
                print
            else:
                print x
    else:
        print pred

def presentPredictionWS2 (pred, output):
    
    """Writes the result of the prediction into a log file and prints some of them in the screen
    """

    with open(output,'w') as fp:
        if pred[0]:
            for compound in pred[1]:  # loop for compounds
                if compound[0]:
                    vaTuple = compound[1][0]
                    adTuple = compound[1][1]
                    riTuple = compound[1][2]
                    
                    fp.write ('%d\t%s\t'%vaTuple)
                    fp.write ('%d\t%s\t'%adTuple)
                    fp.write ('%d\t%s\n'%riTuple)
                else:
                    fp.write ('0\t%s\t0\t0\t0\t0\n' % compound[1])
        else:
            msg = 'ERROR:'+pred[1]
            fp.write ('0\t%s\t0\t0\t0\t0\n' % msg)
                
    

def presentPredictionWS1 (pred):
    
    results = []
    
    if pred[0]:
        #loop for compounds
        for x in pred[1]:
            val = ''
            msg = ''
            stat = 1
            if x[0]:
                y = x[1][0]
                if y[0]:
                    #val = float(y[1])
                    val = y[1]
                    stat = 0
                else:
                    msg = str(y[1])
            else:
                msg = str(x[1])
            results.append((val,stat,msg))
    else:
        stat = 1
        val = ''
        msg = str(pred[1])
        results.append((val,stat,msg))

    pkl = open('results.pkl', 'wb')
    pickle.dump(results, pkl)
    pkl.close()


def presentPredictionS (pred):
    
    pkl = open('results.pkl', 'wb')
    pickle.dump(pred, pkl)
    pkl.close()
    

def presentPrediction (pred, api):

    if   api == 0:                    # 0: command line
                                      # std output in human readable form, 
        presentPredictionText (pred)
                            
    elif api == 1:                    # 1: WS API 1.0 (deprecated)
                                      # pickl file 'results.plk' in cwd
        presentPredictionWS1 (pred)
        
    elif api == 2 or api == 6:        # 2: WS API 2.0
                                      # 6: WS API 3.0
                                      # parseable 'results.txt' in cwd
                                      
        presentPredictionWS2 (pred, 'result.txt')
        
    elif api == 3 or api == 5:        # 3: local models,
                                      # 5: hierarchical models
                                      # pickl file 'results.plk' in cwd
        presentPredictionS (pred)
        
    elif api == 4:                    # eTOXlab GUI
                                      # parseable 'results.txt' in /var/tmp
                                      
        presentPredictionWS2 (pred, '/var/tmp/results.txt')
        

def testimodel():
    try:
        from imodel import imodel
    except:
        return

    print 'please remove file imodel.py or imodel.pyc from eTOXlab/src'
    sys.exit(1)

    
def usage ():
    """Prints in the screen the command syntax and argument"""
    
    print 'predict -e endpoint [-f filename.sdf][-v 1|last][-x]'


def main ():

    endpoint = None
    ver = -99
    loc = -99
    api = 0
    mol = None
    
    progress = False
    detail   = False
    extvalid = False

    try:
       opts, args = getopt.getopt(sys.argv[1:], 'abcge:f:v:s:hqx')

    except getopt.GetoptError:
       writeError('Error. Arguments not recognized')
       usage()
       sys.exit(1)

    if args:
       writeError('Error. Arguments not recognized')
       usage()
       sys.exit(1)
        
    if len( opts ) > 0:
        for opt, arg in opts:

            if opt in '-e':
                endpoint = arg
                
            elif opt in '-f':
                mol = arg
                
            elif opt in '-v':
                if 'last' in arg:
                    ver = -1
                else:
                    try:
                        ver = int(arg)
                    except ValueError:
                        ver = -99

            elif opt in '-x':   ### run external validation using embeeded activity values
                extvalid = True

            ##################################################################
            ###    Internal prediction calls
            ###
            ###    modifier     API         use
            ###     none        0           interactive, command mode use
            ###    -a           1           web service v1 (deprecated)
            ###    -b           2           web service v2 
            ###    -c           6           web service v3 
            ###    -s           3           local model (followed by local ID)
            ###    -g           4           eTOX GUI
            ###    -q           5           internal from hierarchical model
            ###
            ###     Not for use in command mode. Do not document with -h !
            ###
            ##################################################################

            elif opt in '-a':   ### web service call. API v1 (deprecated)
                api = 1
                
            elif opt in '-b':   ### web service call. API v2
                api = 2

            elif opt in '-c':   ### web service call. API v3
                api = 6
                
            elif opt in '-s':   ### call for local models (like HERG4)
                api = 3
                loc = int(arg)

            elif opt in '-g':   ### eTOXlab GUI calls
                api = 4
                
            elif opt in '-q':   ### internal call from hierarchical models (like LQT)
                api = 5

            elif opt in '-h':
                usage()
                sys.exit(0)

    if ver == -99:
        if api in (1,2,5):  # web services API v1 and v2 or hierarchical models do not define versions
            pass
        else:
            usage()
            sys.exit (1)

    if api in (1,2,3,4,5,6):
        sys.path.append ('/opt/RDKit/')
        sys.path.append ('/opt/standardiser/standardise20140206/')

    if api in (2,6):
        progress = True
        
    if not mol:
        if api==0:    # for interactive use the definition of mol is compulsory
            usage()
            sys.exit (1)
        else:         # for non-interactive calls, input_file.sdf is the default
            mol = './input_file.sdf'
        
    if not endpoint:
        usage()
        sys.exit (1)

    # make sure imodel has not been copied to eTOXlab/src. If this were true, this version will
    # be used, instead of those on the versions folder producing hard to track errors and severe
    # misfunction
    
    testimodel()

    result=predict (endpoint,mol,ver,api,loc,
                    detail, progress, extvalid)
    
    presentPrediction (result, api)
    
    sys.exit(0)
        
if __name__ == '__main__':
    
    main()
    
