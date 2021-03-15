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

CONVEYOR_VER = 1    # update only for major changes

class Conveyor:
    ''' Class storing all data generated in the workflows. This class is 
    declared by workflow objects like Build or Predict and passed as an 
    argument to every step

    The class contains data (data) an index (manifest) and an auxiliar 
    dictionary for identifying important data

    It is also used to handle workflow errors. Errors are dumped here and
    checked frequently in the code. When detected, the workflow is aborted

    The class contains method for pickle storage and for convering to JSON
    '''


    def __init__(self):
        ''' constructor '''
        self.conveyor_ver = CONVEYOR_VER
        self.origin = 'undefined'
        self.data = {}
        self.manifest = []
        self.meta = { 'main' : [] }
        self.error = None
        self.warning = None

    def save(self, fo):        
        ''' constructor '''
        pickle.dump(self.conveyor_ver, fo)
        pickle.dump(self.origin, fo)
        pickle.dump(self.data, fo)
        pickle.dump(self.manifest, fo)
        pickle.dump(self.meta, fo)
        pickle.dump(self.error, fo)
        pickle.dump(self.warning, fo)
        
    def load(self, fi):
        ''' constructor '''
        if pickle.load(fi) != self.conveyor_ver:
            return False, 'Wrong version'
        try:
            self.origin = pickle.load(fi)
            self.data = pickle.load(fi)
            self.manifest = pickle.load(fi)
            self.meta = pickle.load(fi)
            self.error = pickle.load(fi)
            self.warning = pickle.load(fi)
        except:
            return False, 'Error extracting pickle'

        return True, 'OK'

    def isKey(self, _key):
        return _key in self.data

    def getOrigin (self):
        return self.origin
    
    def setOrigin (self, value):
        self.origin = value

    def getMain (self):
        return self.meta['main']

    def addMain (self, value):
        self.meta['main'].append(value)

    def addMeta (self, key, value):
        self.meta[key] = value

    def getMeta (self, key):
        if not key in self.meta:
            return None
        return self.meta[key]

    def getError (self):
        return self.error is not None
    
    def getErrorMessage (self):
        return self.error 

    def setError (self, message):
        if self.error is None:
            self.error = str(message)
        else:
            self.error += '\n'+str(message)

    def getWarning (self):
        return self.warning is not None

    def getWarningMessage (self):
        return self.warning

    def setWarning (self, message):
        if self.warning is None:
            self.warning = str(message)
        else:
            self.warning += '\n'+str(message)
        
        print ('warning status:', self.warning)
    
    def getVal(self, key):
        if not key in self.data:
            return None
        return self.data[key]

    def setVal(self, key, value):
        if not key in self.data:
            return
        self.data[key]=value

    def addVal(self, var, _key, _label, _type, _dimension='objs',
               _description=None, _relevance=None):
        '''
        Insert new items at the data dictionary, indexing 
        it appropriatelly in the "manifest" and "meta" 
        
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

    def objectKeys (self):
        ''' returns data keys containing objects values '''
        object_elements = []
        for i in self.manifest:
            if i['dimension'] == 'objs':
                object_elements.append(i['key'])

        return object_elements
    
    def singleKeys (self):
        ''' returns data keys containing single values '''
        single_elements = []
        for i in self.manifest:
            if i['dimension'] == 'single':
                single_elements.append(i['key'])
        return single_elements

    def getJSON (self, xdata=False):
        ''' returns a JSON containing 
            - error/warnings 
            - manifest and meta
            - data (only single and object)
         '''

        temp_json = {}

        if self.error is not None:
            temp_json['error'] = self.error
            return json.dumps(temp_json)

        if self.warning is not None:
            temp_json['warning'] = self.warning

        temp_json['manifest'] = self.manifest
        temp_json['meta'] = self.meta

        white_keys  = self.objectKeys()
        white_keys += self.singleKeys()

        if xdata or (self.getMeta('input_type') == 'model_ensemble') :
            white_keys += ['xmatrix', 'var_nam']

        #print (white_keys)

        for key in white_keys:
            value = self.data[key]

            if key in ['model_build_info', 'model_valid_info']:
                json_temp = []
                for i in value:
                    json_temp.append(self.modelInfoJSON(i))
                temp_json[key] = json_temp

            # np.arrays cannot be serialized to JSON and must be transformed
            elif isinstance(value, np.ndarray):
                temp_json[key]=value.tolist()
            else:
                temp_json[key]=value

        json_result = json.dumps(temp_json, allow_nan=True)
        
        return json_result

    def modelInfoJSON (self,i):
        ''' Results describing the model quality and characteristics are tuples 
            with three elements

            This function returns a version of this tuple suitable for being 
            serialized to JSON
        '''

        # int64
        if 'numpy.int64' in str(type(i[2])):
            try:
                v = int(i[2])
            except:
                v = None
            return((i[0], i[1], v))

        # int64
        if 'numpy.float64' in str(type(i[2])):
            try:
                v = float(i[2])
            except:
                v = None
            return((i[0], i[1], v))

        # ndarrays
        if isinstance(i[2], np.ndarray):
            return((i[0], i[1], i[2].tolist()) )

        return i