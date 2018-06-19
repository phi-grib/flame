#! -*- coding: utf-8 -*-

# Description    Flame Odata class
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
# Copyright 2018 Manuel Pastor
##
# This file is part of Flame
##
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
##
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import json
import numpy as np


class Odata():

    def __init__(self, parameters, results):

        # previous results (eg. object names, molecular descriptors) are retained
        self.results = results
        self.parameters = parameters
        self.format = self.parameters['output_format']

    def _output_md (self):
        ''' dumps the molecular descriptors to a TSV file'''
         
        with open('output_md.tsv', 'w') as fo:

            # Make sure the keys 'var_nam', 'obj_nam', 'xmatrix' actualy exist
            # start writting MD
            if 'var_nam' in self.results:
                # header: obj:name + var name

                header = 'name'
                var_nam = self.results['var_nam']

                for nam in var_nam:
                    header += '\t'+nam
                fo.write(header+'\n')

            if 'xmatrix' in self.results and 'obj_nam' in self.results:
                # extract obj_name and xmatrix
                xmatrix = self.results['xmatrix']
                obj_nam = self.results['obj_nam']

                # iterate for objects
                shape = np.shape(xmatrix)

                if len(shape) > 1:  # 2D matrix (num_obj > 1)
                    for x in range(shape[0]):
                        line = obj_nam[x]
                        for y in range(shape[1]):
                            line += '\t'+str(xmatrix[x, y])
                        fo.write(line+'\n')

                else:             # 1D matrix (num_obj = 1)
                    line = obj_nam[0]
                    for y in range(shape[0]):
                        line += '\t'+str(xmatrix[y])
                    fo.write(line+'\n')

    def run_learn(self):
        ''' Process the results of lear, usually a report on the model quality '''

        if 'model_build' in self.results:
            for val in self.results['model_build']:
                if len(val)<3:
                    print (val)
                else:
                    print (val[0],' (', val[1], ') : ', val[2])

        if 'model_validate' in self.results:
            for val in self.results['model_validate']:
                if len(val)<3:
                    print (val)
                else:
                    print (val[0],' (', val[1], ') : ', val[2])

        # TODO: process learn output and produce meaniningfull JSON/TSV

        if self.parameters['output_md']:
            self._output_md()
            
        return True, 'building OK'

    def run_apply(self):
        ''' Process the results of apply, usually a list of results and serializing to JSON '''
        
        # Check if all mandatory elements are in the results matrix
        
        main_results = self.results['meta']['main']

        for key in main_results:
            if not key in self.results:
                self.results['error'] = 'unable to find "'+key+'" in results'
                return self.run_error()

        output = ''

        if self.parameters['output_md']:
            self._output_md()

        # print ('format output', self.format)

        if 'TSV' in self.format:

            # label and smiles
            key_list = ['obj_nam']
            if 'SMILES' in self.results:
                key_list.append('SMILES')

            # main result
            key_list += self.results['meta']['main']

            # add all object type results
            manifest = self.results['manifest']
            for item in manifest:
                if item['dimension'] == 'objs' and item['key'] not in key_list:
                    key_list.append(item['key'])

            with open('output.tsv', 'w') as fo:
                header = ''
                for label in key_list:
                    header += label+'\t'
                fo.write(header+'\n')

                obj_num = int(self.results['obj_num'])

                for i in range(obj_num):
                    line = ''
                    for key in key_list:

                        if i > len(self.results[key]):
                            val = None
                        else:
                            val = self.results[key][i]

                        if val == None:
                            line += '-'
                        else:
                            if isinstance(val, float):
                                line += "%.4f" % val
                            else:
                                line += str(val)
                        line += '\t'
                    fo.write(line+'\n')

        if 'JSON' in self.format:

            # TODO: output also 'method' keys, like the 'external-validation' or others
            # by setting up at the client side some interface able to show them 
             
            # do not output var arrays, only 'obj' arrays
            black_list = []
            for k in self.results['manifest'] :
                if not (k['dimension'] in ['objs','single']):
                    black_list.append(k['key'])
            
            # print (black_list)

            temp_json = {}

            for key in self.results:

                if key in black_list:
                    continue

                value = self.results[key]

                # print (key, value, type(value))
                
                if 'numpy.ndarray' in str(type(value)):

                    if 'bool_' in str(type(value[0])):
                        temp_json[key] = [
                            'True' if x else 'False' for x in value]
                    else:
                        # this removes NaN and and creates a plain list from ndarrays
                        temp_json[key] = [x if not np.isnan(
                            x) else None for x in value]

                else:
                    temp_json[key] = value

            output = json.dumps(temp_json)

        return True, output

    def run_error(self):
        ''' Formats error messages, sending only the error and the error source '''

        white_list = ['error', 'warning', 'origin']
        error_json = {key: val for key,
                      val in self.results.items() if key in white_list}

        if 'TSV' in self.format:
            with open('error.tsv', 'w') as fo:
                for key, value in error_json.items():
                    fo.write(key+'\t'+value+'\n')

        if 'JSON' in self.format:
            return False, json.dumps(error_json)

        # this is only reached if JSON is not present and
        # expected output is the console
        return False, 'errors found'

    def run(self):
        ''' Formats the results produced by "learn" or "apply" as appropriate '''

        if 'error' in self.results:
            success, results = self.run_error()

        elif self.results['origin'] == 'learn':
            success, results = self.run_learn()

        elif self.results['origin'] == 'apply':

            success, results = self.run_apply()

        else:
            return False, 'invalid result format'

        return success, results
