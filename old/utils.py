# -*- coding: utf-8 -*-

##    Description    eTOXlab utility functions
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

import os
import sys
import shutil
import string
import random
import time
import subprocess
import cPickle as pickle


VERSION = '0.9.7'
wkd = os.path.dirname(os.path.abspath(__file__))

def removefile(file):
    """Removes silently files or whole paths.

       It checks the existence of the file in the argument and if it does not exists do not issue any error
    """
    if not os.path.exists(file):
        return
    if os.path.isfile(file):
        try:
            os.remove(file)
        except OSError:
            pass
    elif os.path.isdir(file):
        try:
            shutil.rmtree( file )
        except OSError :
            pass

def randomName (size=10, chars=string.ascii_uppercase + string.digits):
    """Simple utility for generating random file names

    """
    name = ''
    for i in range (size):
        name+=random.choice(chars)
    return name

def splitSDF (molecules):
    """Splits a SDFile containing 1 to n molecules

       Every molecule is written to a ""m0000000001.sdf"" file and the full list of names is returned in a list
    """

    try:
        finp = open(molecules,'r')
    except:
        return None

    molList = []
    count = 0
    fout = None

    for line in finp:
        if not fout or fout.closed:
            count += 1
            molList.append ("m%0.10d.sdf" % count)
            fout = open(molList[-1], "w")

        fout.write(line)
        
        if "$$$$" in line:
            fout.close()

    finp.close()
    if fout is not None:
        fout.close()

    return molList


def getExternalPrediction (tag, molecules):
    
    removefile ('results.pkl')
    
    call =['/usr/bin/python', wkd+'/predict.py','-e',tag, '-f', molecules, '-q']

    retcode = subprocess.call(call)
    
    if retcode != 0 : return (False, tag +' computation failed to execute')

    try:
        f = open ('results.pkl','rb')
        rtemp = pickle.load(f)
        f.close()
    except:
        return (False, tag+' computation result empty')
            
    if rtemp[0]:
        resulti = rtemp[1][0] # first molecule
        if resulti[0]:
            if resulti[1][0]:
                return (True, resulti[1][0][1])
    
    return (False, tag+' computation failed')


def checkOldSynthax (endpoint):
    """ Backwards compatibility fix for old service-version format. This function creates a new
        file if none has been defined and converts the old format to the new one

        Returns true is successful or false when it fails to create the appropriate file

    """
    
    edir = wkd +'/'+endpoint

    over = -999
    
    try:
        f = open (edir+'/service-version.txt','r')
        line = f.readline()
        f.close

        l = line.split('\t')
        if len (l) == 2:
            return True
           
        over = int(line)
        
    except:
        pass

    t = []
    itemend = os.listdir(edir)
    itemend.sort()

    vi = -99
    for iversion in itemend:
        
        if not os.path.isdir(edir+'/'+iversion): continue
        if not iversion.startswith('version'): continue
        vi = int (iversion[-4:])

        if vi == over:
            t.append ((vi,1))
        else:
            t.append ((vi,0))
    
    try:
        f = open (edir+'/service-version.txt','w')
        for ti in t:
            f.write(str(ti[0])+'\t'+str(ti[1])+'\n')
        f.close()
    except:
        return False

    return True


def lastVersion (endpoint,verID):
    """Returns the path to the directory where the verID version of the model is located

       If a value of -1 is provided, analyzes the "version0000" directories and returns the last one
    """
    epd = wkd+'/'+endpoint

    if not os.path.isdir(epd):
        return None
    
    if verID==-1:       # we request the last version
        v = [int(x[7:]) for x in os.listdir (epd) if x.startswith("version")]
        if not v:
            return None
        verID = max(v)

    epd+='/version%0.4d' % verID

    if not os.path.isdir(epd):
        return None
    else:
        return epd

def exposedVersion (endpoint):
    """Returns the path to the directory defined to hold the web-exposed model or none if
       no such model version has been defined

       FOR BACKWARDS COMPATIBILITY ONLY. The use of this function is not compatible with the new versioning management
       schema and must be reviewed
    """
    epd = wkd+'/'+endpoint

    if not os.path.isdir(epd):
        return None

    if not os.path.isfile (epd+'/service-version.txt'):
        return None

    if not checkOldSynthax (endpoint):
        return None
    
    f = open (epd+'/service-version.txt','r')

    maxPubVer = 0
    maxIntVer = 0

    try:    
        while True:
            line = f.readline()
            if line == '' : break
            l = line.split('\t')
            if int(l[1]) > maxPubVer:
                maxPubVer = int(l[1])
                maxIntVer = int(l[0])
        f.close()
    except:
        return None
    
    if maxIntVer > 0:
        epd+='/version%0.4d' % maxIntVer
    else:
        return None

##    f = open (epd+'/service-version.txt','r')
##    wsID = int(f.readline ())
##    if wsID == 0:
##        return None
##
##    epd+='/version%0.4d' % wsID

    if not os.path.isdir(epd):
        return None
    else:
        return epd

def sandVersion (endpoint):
    """Returns the path to the version0000 directory (sandbox model)

    """
    epd = wkd+'/'+endpoint

    if not os.path.isdir(epd):
        return None

    epd+='/version0000'

    if not os.path.isdir(epd):
        return None
    else:
        return epd
    
def nextVersion (endpoint):
    """Returns the path to the directory where the verID version of the model is located

       If a value of -1 is provided, analyzes the "version0000" directories and returns the last one
    """
    epd = wkd+'/'+endpoint

    if not os.path.isdir(epd):
        return None
    
    v = [int(x[7:]) for x in os.listdir (epd) if x.startswith("version")]
    if not v:
        return None
    verID = max(v)
    verID += 1

    epd+='/version%0.4d' % verID

    return epd

def cleanSandbox (sandDir):

    if not sandDir.endswith('/'):
        sandDir+='/'
        
    files = ['tstruct.sdf',
             'tdata.pkl',
             'info.pkl',
             'ffdexcluded.pkl',
             'view-property.pkl',
             'view-model-pca.pkl',
             'view-background-pca.txt',
             'view-background-property.txt']
        
    for f in files:
        removefile (sandDir+f)

def updateProgress(progress):
    """Prints a progress bar in the screen.

       Progress must be specificed as a number from 0 to 1 
    """
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0.0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0.0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1.0
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100.0, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def writeError (error, verbose=False):
    """Print an error message"""

    if verbose:
        print error
        
    try:
        f=open('./error.log','a+')
    except:
        return
    
    f.write (time.strftime('%c')+' : '+error+'\n')
    f.close()


