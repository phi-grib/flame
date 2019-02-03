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

import pickle
import numpy as np
import json
from flame.util import utils

class Conveyor:
    ''' Class storing ***

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

    def save(self,fo):
        pickle.dump(self.results_format, fo)
        pickle.dump(self.origin, fo)
        pickle.dump(self.data, fo)
        pickle.dump(self.manifest, fo)
        pickle.dump(self.meta, fo)
        pickle.dump(self.error, fo)
        pickle.dump(self.warning, fo)
        
    def load(self, fi):
        self.results_format = pickle.load(fi)
        self.origin = pickle.load(fi)
        self.data = pickle.load(fi)
        self.manifest = pickle.load(fi)
        self.meta = pickle.load(fi)
        self.error = pickle.load(fi)
        self.warning = pickle.load(fi)

    def getVal(self, key):
        if not key in self.data:
            return None
        return self.data[key]

    def getError (self):
        return self.error is not None
    
    def getWarning (self):
        return self.warning is not None

    def getErrorMessage (self):
        return self.error 

    def getWarningMessage (self):
        return self.warning

    def setError (self, message):
        self.error = message

    def setWarning (self, message):
        self.warning = message
    
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
    
    def singleKeys (self):
        single_elements = []
        for i in self.manifest:
            if i['dimension'] == 'single':
                single_elements.append(i['key'])
        return single_elements

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

    def getOrigin (self):
        return self.origin
    
    def setOrigin (self, value):
        self.origin = value

    def getJSON (self):

        temp_json = {}

        if self.error is not None:
            temp_json['error']
            return json.dumps(temp_json)

        if self.warning is not None:
            temp_json['warning']

        temp_json['manifest'] = self.manifest
        temp_json['meta'] = self.meta

        white_keys  = self.objectKeys()
        white_keys += self.singleKeys()

        for key in white_keys:
            value = self.data[key]

            if key in ['model_build_info', 'model_valid_info']:
                json_temp = []
                for i in value:
                    json_temp.append(utils.results_info_to_JSON(i))
                temp_json[key] = json_temp

            # np.arrays cannot be serialized to JSON and must be transformed
            elif isinstance(value, np.ndarray):
                temp_json[key]=value.tolist()
            else:
                temp_json[key]=value

        print (json.dumps(temp_json))
        
        return json.dumps(temp_json)
