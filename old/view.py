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
import shutil
import subprocess

from utils import lastVersion
from utils import writeError
from utils import removefile

def view (endpoint, molecules, verID, vtype, background, refname, refver):
    """Top level view function

       molecules:  SDFile containing the collection of 2D structures to be predicted
       verID:      version of the model that will be used. Value -1 means the last one
       
    """
    if verID != -99:
        va = lastVersion (endpoint, verID)

##    if not molecules:
##        molecules = va+'/training.sdf'

    # load model
    try:
        sys.path.append(va)
        from imodel import imodel
        model = imodel (va)
    except:
        return (False, 'unable to load imodel')

    if not model:
        return (False, 'unable to load imodel')

    # arguments of the call overwrite existing view settings of imodel.py


    ## viewMode = query or series
    if vtype != None :
        model.viewType = vtype

    if background != None:
        model.viewBackground = background

    if molecules:
        model.viewMode = 'query'
        model.viewReferenceEndpoint = endpoint
        model.viewReferenceVersion = verID
    else:
        model.viewMode = 'series'
        if refname != None:
            model.viewReferenceEndpoint = refname

        if refver != None:
            model.viewReferenceVersion = refver
        
    result = model.viewWorkflow (molecules)

    return (result)

def presentResults (result):
    """Writes the result of the model building
    """

    if not result[0]:
        print '\nERROR:', result[1]
        sys.stdout.flush()
        sys.exit(1)
        
    for i in result[1]:
        print i

    sys.exit(0)

    
def usage ():
    """Prints in the screen the command syntax and argument"""
    
    print 'view -e endpoint [-f filename.sdf][-v 1|last][--type=pca|property][--background][--refname=refname][--refver=0]'

def main ():

    endpoint = None
    ver = -99
    mol = None
    vtype = None
    background = None
    refname = None
    refver = None

    try:
       opts, args = getopt.getopt(sys.argv[1:], 'e:f:v:h', ['type=', 'background', 'refname=', 'refver='])

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

            elif opt in '-h':
                usage()
                sys.exit(0)
            elif opt in '--type':
                vtype = arg
            elif opt in '--background':
                background = True
            elif opt in '--refname':
                refname = arg
            elif opt in '--refver':
                try:
                    refver = int(arg)
                except:
                    refver = 0

    if not mol and ver==-99:
        usage()
        sys.exit(1)
        
    if not endpoint:
        usage()
        sys.exit (1)

    if vtype not in [None, 'pca','property', 'project', 'model']:
        print '+',property,'+'
        usage()
        sys.exit (1)

    if vtype == 'project' and (refname == None and mol == None):
        print 'project view type requires to define the reference endpoint name'
        usage()
        sys.exit (1)

    if vtype == 'project' and refver == None:
        refver = 0
        
##    print vtype, background, refname, refver
    
    result=view (endpoint, mol, ver, vtype, background, refname, refver)

    presentResults (result)

        
if __name__ == '__main__':
    
    main()
