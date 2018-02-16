#!/usr/bin/env python
# -*- coding: utf-8 -*-

##    Description    eTOXlab component for creating a new predictive model
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
import shutil

from utils import *
  
def build (endpoint, molecules, model, verID, loc):
    """Top level buildind function

       molecules:  SDFile containing the collection of 2D structures to be predicted
       verID:      version of the model that will be used. Value -1 means the last one

    """
    
    if (verID!=None):
        vv = lastVersion (endpoint, verID)  # full path to endpoint+version or last if -1 is provided
        
    va = sandVersion (endpoint)

    if loc!=None:
        va += '/local%0.4d' % loc

    # copy training set to sandbox, either from argument or from version
    if molecules:

        cleanSandbox(va)
        
        try:
            shutil.copy (molecules,va+'/training.sdf')
        except:
            return (False, 'file:'+molecules+' not found')
    else:
        
        if vv != va:

            cleanSandbox(va)
            
            files = ['/training.sdf',
                     '/tstruct.sdf',
                     '/tdata.pkl']
            for i in files:
                if os.path.isfile(vv+i):
                    shutil.copy(vv+i,va)
                    
            ##shutil.copy (vv+'/training.sdf',va)

    # copy model to sandbox, either from argument or from version
    if model:
        shutil.copy (model,va+'/imodel.py')
    else:
        if vv != va:
            shutil.copy (vv+'/imodel.py',va)
    
    # load model
    try:
        sys.path.append(va)
        from imodel import imodel
        model = imodel (va)
    except:
        return (False, 'unable to load imodel')

    if not model:
        return (False, 'unable to load imodel')
##        sys.path.append(va)
##        from imodel import imodel
##        model = imodel (va)
    
    result = model.buildWorkflow(molecules)

##    if not model.buildable:
##        success, result = model.log ()
##        if not success:
##            return (False, result)
##        return (result)
##    
##    # load data, if stored, or compute it from the provided SDFile
##
##    dataReady = False
##    
##    if not molecules:
##        dataReady = model.loadData ()
##        
##        if not model.loadSeriesInfo ():
##            model.setSeries ('training.sdf', len(model.tdata))  
##
##    if not dataReady: # datList was not completed because load failed or new series was set
##
##        # estimate number of molecules inside the SDFile
##        nmol = 0
##        try:
##            f = open (va+'/training.sdf','r')
##        except:
##            return (False,"Unable to open file %s" % molecules)
##        for line in f:
##            if '$$$$' in line: nmol+=1
##        f.close()
##
##        if not nmol:
##            return (False,"No molecule found in %s:  SDFile format not recognized" % molecules)
##
##        model.setSeries (molecules, nmol)
##        
##        i = 0
##        fout = None
##        mol = ''
##        
##        # open SDFfile and iterate for every molecule
##        f = open (va+'/training.sdf','r')
##
##        updateProgress (0.0)
##        
##        for line in f:
##            if not fout or fout.closed:
##                i += 1
##                mol = 'm%0.10d.sdf' % i
##                fout = open(mol, 'w')
##
##            fout.write(line)
##        
##            if '$$$$' in line:
##                fout.close()
##
##                ## workflow for molecule i (mol) ############
##                success, result = model.normalize (mol)
##                if not success:
##                   writeError('error in normalize: '+result)
##                   continue
##
##                molFile   = result[0]
##                molName   = result[1]
##                molCharge = result[2]
##                molPos    = model.saveNormalizedMol(molFile)
##                
##                success, infN = model.extract (molFile,molName,molCharge,molPos)
##                if not success:
##                   writeError('error in extract: '+ str(infN))
##                   continue
##
##                updateProgress (float(i)/float(nmol))
##                ##############################################
##
##                removefile (mol)
##
##        f.close()
##        if fout :
##            fout.close()
##
##        model.saveData ()
##
##    # build the model with the datList stored data
##    
##    success, result = model.build ()
##    if not success:
##        return (False, result)
##
##    success, result = model.log ()
##    if not success:
##        return (False, result)

    return (result)

def presentResults (result):
    """Writes the result of the model building
    """
    #print result

    if not result[0]:
        print '\nERROR:', result[1]
        sys.stdout.flush()
        sys.exit(1)
    else:
        print result

    sys.exit(0)

def testimodel():
    try:
        from imodel import imodel
    except:
        return

    print 'please remove file imodel.py or imodel.pyc from eTOXlab/src'
    sys.exit(1)

def usage ():
    """Prints in the screen the command syntax and argument"""
    
    print 'ERROR: build -e endpoint [-f filename.sdf][-m model.py][-v 1|last]'

def main ():

    endpoint = None
    ver = None
    mol = None
    mod = None
    loc = None

    try:
       opts, args = getopt.getopt(sys.argv[1:], 'e:f:m:v:s:h')

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
            elif opt in '-m':
                mod = arg
            elif opt in '-v':
                if 'last' in arg:
                    ver = -1
                else:
                    try:
                        ver = int(arg)
                    except ValueError:
                        ver = None
            elif opt in '-s':
                loc = int(arg)
            elif opt in '-h':
                usage()
                sys.exit(0)

    if (mol==None) and (ver==None):
        usage()
        sys.exit(1)

    if (mod==None) and (ver==None):
        usage()
        sys.exit(1)

    if (mod!=None) and (mol!=None) and (ver!=None):
        usage()
        sys.exit(1)
       
    # make sure imodel has not been copied to eTOXlab/src. If this were true, this version will
    # be used, instead of those on the versions folder producing hard to track errors and severe
    # misfunction
    testimodel()
    
    result=build (endpoint,mol,mod,ver,loc)

    if loc==None:
        presentResults (result)

    sys.exit(0)
        
if __name__ == '__main__':
    
    main()
