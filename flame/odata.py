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

class Odata():

    def __init__ (self, parameters, results, out_format='JSON' ):

        self.results = results
        self.parameters = parameters
        self.format = out_format


    def run_learn (self):

        print ('odata : ', self.results)
        print ('building OK!')
        return True, 'ok'


    def run_apply (self):

        # names and structures
        # JSON serialization (if out_format is JSON)

        #print ('odata : ', self.results)

        if not 'projection' in self.results:
            return False, self.results
        
        # numpy arrays must be converted to lists before they
        # can be serialized by json.dumps
        temp_json = {}
        if not self.parameters['conformal']:
            
            if self.parameters['quantitative']:
                temp_json = {
                    'obj_nam': self.results['obj_nam'],
                    'projection': self.results['projection'].tolist(),
                    'CI': self.results['CI'].tolist(),
                    'RI': self.results['RI'].tolist()}
            else:
                temp_json = {
                    'obj_nam': self.results['obj_nam'],
                    'projection': self.results['projection'].tolist(),
                    'CI': self.results['CI'].tolist(),
                    'RI': self.results['RI'].tolist()}
                
        else:
            if self.parameters['quantitative']:
                temp_json = {
                    'obj_nam': self.results['obj_nam'],
                    'projection': self.results['projection']['values'].tolist(),
                    'lower_limit': self.results['projection']['lower_limit'].tolist(),
                    'upper_limit': self.results['projection']['upper_limit'].tolist(),
                    'CI': self.results['CI'].tolist(),
                    'RI': self.results['RI'].tolist()}
            else:
                temp_json = {
                    'obj_nam': self.results['obj_nam'],
                    'projection': self.results['projection'],
                    'CI': self.results['CI'].tolist(),
                    'RI': self.results['RI'].tolist()}
            


        return True, json.dumps(temp_json) 


    def run (self):

        if not 'origin' in self.results:
            return False, 'invalid result format'

        if self.results['origin'] == 'learn':
            success, results  = self.run_learn ()

        elif self.results['origin'] == 'apply':
            success, results  = self.run_apply ()

        else:
            return False, 'invalid result format'

        return success, results

        

