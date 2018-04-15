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
import sys
import yaml

def read_configuration ():
    conf = {}
    source_dir = os.path.dirname(os.path.abspath(__file__))[:-5] #removing '/utils'

    with open (source_dir+'/config.yaml', 'r') as config_file:
        conf = yaml.load(config_file)

    ## if the name of a path starts with '.' we will prepend the path with the source dir
    if conf ['model_repository_path'][0] =='.' :
        conf ['model_repository_path']=source_dir+conf ['model_repository_path'][1:]

    print (conf)

    #conf ['model_repository_path'] = source_dir + '/models'
    #conf ['model_repository_path'] = 'C:/flame/models'
    return conf
    
## read configuraton file and store in a variable to prevent reading files more
## than strictly necessary
configuration = read_configuration ()

def model_repository_path ():
    return configuration ['model_repository_path']

def model_tree_path (model):
    return model_repository_path()+'/'+model

def model_path (model, version):        
       
    modpath = model_tree_path(model)
    
    if version == 0 :
       modpath += '/dev'
    else:
       modpath += '/ver%0.6d'%(version)
    
    return modpath

def module_path (model, version):

    modreppath = model_repository_path()
    if not modreppath in sys.path:
        sys.path.insert(0,modreppath)

    print (sys.path)

    ##modpath = 'models'+'.'+model
    modpath = model

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

def intver(raw_version):
    if raw_version is None:
        return 0
    
    try:
        version = int(raw_version)
    except:
        version = 0

    return version
