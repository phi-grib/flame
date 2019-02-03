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

# import os
# import yaml
# import json

# from flame.util import utils

class Conveyor:
    ''' Class storing a large set of parameters defining how a model is built

        These parameters are loaded from a configuration file (typically 
        in yaml format) 

        The version 1 of parameters.yaml is a simple "key-value" python dictionary
        in yaml file

        In version 2 every parameter is a dictionary with keys defining the parameter type, 
        value and providing a human-readable explanation used for the GUI

        This code supports both versions of the parameter file, but the use of version 1
        is deprecated and will not be supported indefinitely 
    '''

    def __init__(self):
        ''' constructor '''
        self.results_format = 1
        self.origin = 'undefined'
        self.data = {}
        self.manifest = []
        self.meta = { 'main' : [] }
        self.error = None
        self.warning = None

    def getVal(self, key):
        if not key in self.data:
            return None
        return self.data[key]

    def getError (self):
        return self.error is not None
    
    def getWarning (self):
        return self.error is not None

    def getErrorMessage (self):
        return self.error 

    def getWarningMessage (self):
        return self.message 

    def setError (self, message):
        self.error = 'message'

    def setWarning (self, message):
        self.warning = 'message'
    
    def isKey(self, _key):
        return _key in self.data

    def setVal(self, key, value):
        if not key in self.data:
            return
        self.data[key]=value

    def objectKeys (self):
        object_elements = []
        for i in self.manifest:
            if i['dimension'] == 'objs':
                object_elements.append(i['key'])

        return object_elements

    def addVal(self, var, _key, _label, _type, _dimension='objs',
               _description=None, _relevance=None):
        '''
        Utility function to insert information within the "result" dictionary, indexing 
        it appropriatelly in the "manifest" and "meta" keys
        
        _key         (str)
                    key for including this data in results dictionary
        
        _label       (str) 
                    descriptive text used to label this data in tables
        
        _type        [label | decoration | smiles | result | confidence | method]
                    cathegory of data, used to decide how showing it in GUI's
        
        _dimension   [single | vars | objs]
                    if the data is one/more isolated values (single) or an array
                    with data for each X variable (vars) or object (objs) 
        
        _description (str)
                        a long human readable description of the information
        
        _relevance   [main | None]
                        the main label is asigned to data to be highlighed 
        '''
        
        # add the data 
        self.data[_key] = var

        # insert the information in manifest
        manifest_item = {'key': _key,
                        'label': _label,
                        'type': _type,
                        'dimension': _dimension,
                        'description': _description,
                        'relevance': _relevance
                        }
    
        self.manifest.append(manifest_item)

        # if the are adding data of type 'main' insert the key in main
        if _relevance == 'main':
            self.addMain(_key)

    def addMeta (self, key, value):
        self.meta[key] = value

    def addMain (self, value):
        self.meta['main'].append(value)

    def getMain (self):
        return self.meta['main']

    def getJSON (self):
        # TODO
        return ''