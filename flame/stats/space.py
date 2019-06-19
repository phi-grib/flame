#! -*- coding: utf-8 -*-

# Description    Flame Parent Space Class
##
# Authors:       Manuel Pastor (manuel.pastor@ufp.edu)
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
import pickle
import numpy as np
import os
from flame.util import utils, get_logger, supress_log

LOG = get_logger(__name__)


class Space:
    def __init__(self, X, parameters):
        """Initializes the estimator.
        Actions
        -------
            - Attribute assignment
        """

    def build(self):
        ''' This function saves estimator and scaler in a pickle file '''

        # fingeprints must be stored in a list
        # md=MACCSkeys.GenMACCSKeys(mi)

        return True, 'success'

    def search (self):

        # def computePredictionOther (self, md, charge,molSmi):
        #         # empty method to be overriden
        #     if self.model != 'fp-maccs-tanimoto':
        #             return (False, 'not implemented')

        #     nrNeighbours = self.nrMaxNeighbours 
        #     threshold =    self.tanimotoThreshold
        #     dist = []
        #     ids  = []
        #     smis  = []
        #     names = []
        #     # get internal ID and calculate distance to FP
        #         for i in self.tdata:
        #         d = DataStructs.FingerprintSimilarity(md,i[2], metric=DataStructs.TanimotoSimilarity)
        #         if( d > threshold ):
        #                 dist.append( d )
        #             ids.append(i[6])
        #             smis.append(i[7])
        #             names.append(i[8])

        #     sortDist = np.sort( np.array(zip(ids,dist,smis,names),dtype=[('ids', 'S10'), ('dist', float),('smis', 'S1000' ),('names', 'S100' )]), order="dist")
            
        #     js = []
        #     js.append( { "STRUCTURE": molSmi })
            
        #     for j in range( 0, min( nrNeighbours , len(sortDist) )):
        #         m = sortDist[len(sortDist)-j-1]
        #         js.append( {  "STRUCTURE": m[2]})
        #         js.append( {"names": m[3] ,"ID" : m[0] , "DIST" : str(m[1]), 
        #                             })

        #         result = json.JSONEncoder().encode(js)
                    
        #         return result

    def save_space(self):
        ''' This function saves estimator and scaler in a pickle file '''


        return

    def load_space(self):
        ''' This function loads estimator and scaler in a pickle file '''
    
        return