#! -*- coding: utf-8 -*-

# Description    Flame Learn class
#
# Authors: Manuel Pastor (manuel.pastor@upf.edu),
#          Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.

from flame.stats.space import Space
from flame.util import utils, get_logger
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import RobustScaler
import pickle
import os

LOG = get_logger(__name__)

class Slearn:

    def __init__(self, parameters, conveyor):

        self.param = parameters
        self.conveyor = conveyor
        self.conveyor.setOrigin('slearn')

        self.X = self.conveyor.getVal('xmatrix')

    def preprocess(self):
        ''' 
        This function scales the X matrix and selects features 
        The scaler and the variable mask are saved in a pickl file 
        '''
 
        self.scaler = None
        
        # update if other fingerprints are added
        isFingerprint = (self.param.getVal('computeMD_method') == ['morganFP'])

        # Run scaling for MD but never for fingerprints
        if self.param.getVal('modelAutoscaling'):
            if isFingerprint:
                LOG.info('fingerprint descriptors are not scaled for similarity')
            else:
                try:
                    scaler = None
                    if self.param.getVal('modelAutoscaling') == 'StandardScaler':
                        scaler = StandardScaler()
                        LOG.info('Data scaled using StandarScaler')

                    elif self.param.getVal('modelAutoscaling') == 'MinMaxScaler':
                        scaler = MinMaxScaler(copy=True, feature_range=(0,1))
                        LOG.info('Data scaled using MinMaxScaler')

                    elif self.param.getVal('modelAutoscaling') == 'RobustScaler':
                        scaler = RobustScaler()
                        LOG.info('Data scaled using RobustScaler')

                    else:
                        return False, 'Scaler not recognized'

                    if scaler is not None:
                        # The scaler is saved so it can be used later
                        # to prediction instances.
                        self.scaler = scaler.fit(self.X)

                        # Scale the data.
                        self.X = scaler.transform(self.X)

                except Exception as e:
                    return False, f'Unable to perform scaling with exception: {e}'
          

        # This dictionary contain all the objects which will be needed
        # for prediction
        prepro = {'scaler':self.scaler,\
                  'version':1}

        prepro_pkl_path = os.path.join(self.param.getVal('model_path'),
                                      'preprocessing.pkl')
        
        with open(prepro_pkl_path, 'wb') as handle:
            pickle.dump(prepro, handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)

        return True, 'OK'


    def run (self):
        '''
        Builds a chemical space 

        '''

        # pre-process data
        success, message = self.preprocess()
        if not success:
            self.conveyor.setError(message)
            return

        # instances space object
        space = Space(self.param)

        # builds space from idata results
        LOG.debug('Starting space building')

        objinfo = {}
        itemlist = ['obj_nam', 'obj_id', 'SMILES', 'ymatrix']
        for item in itemlist:
            item_val = self.conveyor.getVal(item)
            if item_val is not None:
                objinfo [item] = item_val

        success, space_building_results = space.build(self.X, objinfo)
        if not success:
            LOG.error('space_building_results')
            self.conveyor.setError(space_building_results)
            return

        self.conveyor.addVal(
                    space_building_results,
                    'space_build_info',
                    'space build info',
                    'method',
                    'single',
                    'Information about the building of the chemical space')

        # save model
        try:
            space.save_space()
        except Exception as e:
            LOG.error(f'Error saving space with exception {e}')
            self.conveyor.setError(f'Error saving space with exception {e}')
            return

        LOG.info('Space building finished successfully')

        return

