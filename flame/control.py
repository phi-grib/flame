#! -*- coding: utf-8 -*-

# Description    Flame Control class
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

from flame.util import utils
import os
import yaml


class Control:

    def __init__(self, model, version):

        self.yaml_file = utils.model_path(model, version) + '/parameters.yaml'

        success, parameters = self.load_parameters(model)

        # TODO: study the pros and cons of copying the
        # children template instead
        if not success:
            print('CRITICAL ERROR: unable to load parameter file.'
                  'Running with fallback defaults')
            parameters = self.get_defaults()

        self.parameters = parameters
        self.parameters['endpoint'] = model
        self.parameters['version'] = version
        self.parameters['model_path'] = utils.model_path(model, version)
        self.parameters['md5'] = utils.md5sum(self.yaml_file)

    def load_parameters(self, model):
        '''
        Loads parameters from a yaml file
        '''

        if not os.path.isfile(self.yaml_file):
            return False, None

        try:
            with open(self.yaml_file, 'r') as pfile:
                parameters = yaml.load(pfile)
        except:
            return False, None

        return True, parameters

    # def save_parameters (self, parameters):
    #     yaml.dump(open(self.yaml_file,'w'), parameters)

    def get_parameters(self):
        '''
        Commodity function to access stored parameters
        '''

        return self.parameters

    def get_model_set(self):
        '''
        Returns a Boolean indicating if the model uses external input
        sources and a list with these sources.
        '''

        ext_input = False
        model_set = None

        if 'ext_input' in self.parameters:
            if self.parameters['ext_input']:
                if 'model_set' in self.parameters:
                    if len(self.parameters['model_set']) > 1:
                        model_set = self.parameters['model_set']
                        ext_input = True

        return ext_input, model_set

    def get_defaults(self):
        '''
        Fallback for setting parameters even when
        no "config.yaml" file is found
        '''

        self.yaml_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'children', 'parameters.yaml')

        with open(self.yaml_file, 'r') as pfile:
            parameters = yaml.load(pfile)

        return parameters
