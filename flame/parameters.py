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
import sys
import yaml

from flame.util import utils, get_logger

LOG = get_logger(__name__)

class Parameters:

        def __init__(self, model, version):
     
            parameters_file_path = utils.model_path(model, version)
            parameters_file_name = os.path.join (parameters_file_path,'parameters.yaml')

            if not os.path.isfile(parameters_file_name):
                LOG.critical('Unable to load model parameters. Aborting...')
                sys.exit()

            try:
                with open(parameters_file_name, 'r') as pfile:
                    self.p = yaml.load(pfile)
            except Exception as e:
                LOG.critical('Unable to load model parameters. Aborting...')
                sys.exit()

            self.setVal('endpoint',model)
            self.setVal('version',version)
            self.setVal('model_path',parameters_file_path)
            self.setVal('md5',utils.md5sum(parameters_file_name))

            return

        def getVal(self, key):
            ''' Return the value of the key parameter or None if it is
                not found in the parameters dictionary
            ''' 
            if key in self.p:
                return self.p[key]
            return None

        def getOldParam(self):
            ''' Returns the dictionary with the parameters
                This function was defined only for compatibility purposes
                during the implementation of this class
            '''
            return self.p

        def setVal(self, key, value):
            ''' Sets the parameter defined by key to the given value
            '''
            self.p[key] = value

        def appVal(self, key, value):
            ''' Appends value to the end of existing key list 
            '''
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