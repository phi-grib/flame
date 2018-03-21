#! -*- coding: utf-8 -*-

##    Description    Misc tools 
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
            
import hashlib
import os

def md5sum(filename, blocksize=65536):

    hash = hashlib.md5()

    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)

    return hash.hexdigest()

def base_path (model):

    wkd = os.path.dirname(os.path.abspath(__file__))
    wkd = wkd[:-5] # remove '/utils'
    
    epd = wkd+'/models/'+model

    return epd

def model_path (model, version):        
    
    wkd = os.path.dirname(os.path.abspath(__file__))
    wkd = wkd[:-5] # remove '/utils'
    
    epd = wkd+'/models/'+model
    if version == 0 :
        epd += '/dev'
    else:
        epd += '/ver%0.6d'%(version)
    
    return epd