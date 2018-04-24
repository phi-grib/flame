#! -*- coding: utf-8 -*-

##    Description    Flame Odata class
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
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import json
import numpy as np

class Odata():

    def __init__ (self, parameters, results ):

        self.results = results # previous results (eg. object names, molecular descriptors) are retained 
        self.parameters = parameters
        self.format = self.parameters['output_format']


    def run_learn (self):
        ''' Process the results of lear, usually a report on the model quality '''

        print ('building OK!')
        return True, 'ok'


    def run_apply (self):
        ''' Process the results of apply, usually a list of results and serializing to JSON '''

        ## Check if all mandatory elements are in the results matrix
        main_results = self.results['meta']['main']
        
        for key in main_results:
            if not key in self.results:
                self.results['error'] = 'unable to find "'+key+'" in results'
                return self.run_error()
        
        if self.format=='JSON':
            ## do not output var arrays, only obj arrays
            ## TODO: replace this hardcoded list with var type from manifest
            black_list = ['xmatrix', 'confidence', 'var_nam', 'conf_nam']   

            temp_json = {}

            for key in self.results:

                if key in black_list :
                    continue

                value = self.results[key]

                if 'numpy.ndarray' in str(type(value)):
                    
                    if 'bool_' in str(type(value[0])):
                        temp_json[key] = ['True' if x else 'False' for x in value]
                    else:
                        # this removes NaN and and creates a plain list from ndarrays
                        temp_json[key] = [x if not np.isnan(x) else None for x in value]

                else:
                    temp_json[key]=value

            output = json.dumps(temp_json)

        elif self.format == 'TSV':
            output = 'not implemented'
            
        return True, output


    def run_error (self):
        ''' Formats error messages, sending only the error and the error source '''
        
        white_list = ['error', 'origin']
        error_json = { key: val for key, val in self.results.items() if key in white_list } 

        if self.format=='JSON':
            return True, json.dumps(error_json)    
        
        if self.format == 'TSV':
            return False, 'not implemented'


    def run (self):
        ''' Formats the results produced by "learn" or "apply" as appropriate '''

        if 'error' in self.results:
            success, results = self.run_error ()

        elif self.results['origin'] == 'learn':
            success, results  = self.run_learn ()

        elif self.results['origin'] == 'apply':

            success, results  = self.run_apply ()

        else:
            return False, 'invalid result format'

        return success, results

        