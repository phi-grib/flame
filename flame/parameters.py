#! -*- coding: utf-8 -*-

# Description    Flame Build class
#
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
#
# Copyright 2018 Manuel Pastor
#
# This file is part of Flame
#
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
#
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import yaml
import json

from flame.util import utils

class Parameters:
    ''' Class storing a large set of parameters defining how a model is built

        These parameters are loaded from a configuration file (typically 
        in yaml format) 

        Every parameter is a dictionary with keys defining the parameter type, 
        value and providing a human-readable explanation used for the GUI
    '''

    def __init__(self):
        ''' constructor '''
        self.extended = False
        self.version = 1

        return

    # def loadDict (self, d):
    #     ''' load the content from a dictionary '''
    #     self.p = d    
    #     return

    def loadYaml (self, model, version):       
        ''' load a set of parameters from the configuration file present 
            at the model directory

            adds some parameters identifying the model and the 
            hash of the configuration file 
        '''
        parameters_file_path = utils.model_path(model, version)
        parameters_file_name = os.path.join (parameters_file_path,
                                            'parameters.yaml')

        if not os.path.isfile(parameters_file_name):
            return False

        try:
            with open(parameters_file_name, 'r') as pfile:
                self.p = yaml.load(pfile)
        except Exception as e:
            return False

        if 'version' in self.p:
            self.extended = True
            self.version = self.getVal('version')
        else:
            self.extended = False
            self.version = 1

        self.setVal('endpoint',model)
        self.setVal('version',version)
        self.setVal('model_path',parameters_file_path)
        self.setVal('md5',utils.md5sum(parameters_file_name))

        return True

    @staticmethod
    def saveYaml(self, model, version, input_JSON):
        p = json.load(input_JSON)
        parameters_file_path = utils.model_path(model, version)
        parameters_file_name = os.path.join (parameters_file_path,
                                            'parameters.yaml')
        try:
            with open(parameters_file_name, 'w') as pfile:
                yaml.dump (p, pfile)
        except Exception as e:
            return False

        return True

    def getVal(self, key):
        ''' Return the value of the key parameter or None if it is
            not found in the parameters dictionary
        ''' 
        if not key in self.p:
            return None

        if self.extended:
            if 'value' in self.p[key]:
                return self.p[key]['value']
            return None
        else:
            return self.p[key]

    
    def getDict(self, key):
        ''' Return the value of the key parameter or None if it is
            not found in the parameters dictionary
        ''' 
        if self.extended:
            d = {}
            if key in self.p:
                element = self.p[key]['value']
                if isinstance(element ,dict):
                    for k, v in element.items():
                        if 'value' in v:
                            d[k] = v['value']
            return d
        else:
            return self.p[key]
    
    # def getOldParam(self):
    #     ''' Returns the dictionary with the parameters
    #         This function was defined only for compatibility purposes
    #         during the implementation of this class
    #     '''
    #     return self.p

    def setVal(self, key, value):
        ''' Sets the parameter defined by key to the given value
        '''
        if self.extended:
            if key in self.p:
                if "value" in self.p[key]:
                    self.p[key]["value"] = value
            else:
                self.p[key] = {'value': value}
        else:
            self.p[key]=value

    def appVal(self, key, value):
        ''' Appends value to the end of existing key list 
        '''
        if self.extended:
            if key in self.p:
                if "value" in self.p[key]:
                    self.p[key]['value'].append(value)
        else:
            self.p[key].append(value)

    def getModelSet (self):
        ''' Returns a Boolean indicating if the model uses external input
            sources and a list with these sources 
        '''
        ext_input = False
        model_set = None

        if self.getVal('ext_input'):
            model_set = self.getVal('model_set')
            if model_set is not None:
                if len(model_set) > 1:
                    ext_input = True

        return ext_input, model_set

    def dumpJSON (self):
        return json.dumps(self.p)

