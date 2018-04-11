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

working_dir = os.path.dirname(os.path.abspath(__file__))[:-5] #removing '/utils'

root_dir = working_dir + '/models' 

def root_path ():
    return root_dir

def base_path (model):
    return root_dir+'/'+model

def model_path (model, version):        
       
    epd = base_path(model)
    
    if version == 0 :
        epd += '/dev'
    else:
        epd += '/ver%0.6d'%(version)
    
    return epd

def module_path (model, version):

    modpath = 'models'+'.'+model

    if version == 0 :
        modpath += '.dev'
    else:
        modpath += '.ver%0.6d'%(version)

    return modpath


def md5sum(filename, blocksize=65536):

    hash = hashlib.md5()

    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)

    return hash.hexdigest()