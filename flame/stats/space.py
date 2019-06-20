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


import pickle
import numpy as np
from scipy.spatial import distance 
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from flame.util import utils, get_logger, supress_log

LOG = get_logger(__name__)


class Space:
    def __init__(self, param):
        """Initializes the chemical space

        """
        self.param = param

    def build(self, X, names, SMILES):
        ''' This function saves estimator and scaler in a pickle file '''

        self.names = names
        self.SMILES = SMILES
        self.nobj, self.nvarx = np.shape(X)

        # if X contains fingerprints as numpy, convert to RDKit BitVector to speed-up
        # future similarity searches

        if self.param.getVal('computeMD_method') == ['morganFP']:
            self.X = []
            for i in X:
                bitestring="".join(i.astype(str))
                self.X.append(DataStructs.cDataStructs.CreateFromBitString(bitestring))
        else:
            self.X = X

        print (self.nobj, self.nvarx)
        print (self.names[0], self.SMILES[0], self.X[0])

        return True, 'success'

    def search (self, X, cutoff, numsel):

        # load pickle with reference space
        self.load_space()

        if cutoff is None:
            cutoff = 0.0

        selected_i = []
        selected_d = []
        maxd = 0.0
        maxi = 0


        if self.param.getVal('computeMD_method') == ['morganFP']:
            for i, inpvector in enumerate(X):
                bitestring="".join(inpvector.astype(str))
                ivector = DataStructs.cDataStructs.CreateFromBitString(bitestring)

                for j, jvector in enumerate(self.X):
                    d = DataStructs.FingerprintSimilarity(ivector,jvector, metric=DataStructs.TanimotoSimilarity)
                    
                    if d <= cutoff:
                        continue

                    if len(selected_i) < numsel:
                        selected_i.append(j)
                        selected_d.append(d)
                        if d > maxd:
                            maxd = d
                            maxi = j
                    else:
                        if d <= maxd:
                            continue
                        selected_i[maxi]=j
                        selected_d[maxi]=d
                        maxd = d
                        maxi = j


            for i in range(len(selected_i)):
                print (selected_d[i], self.names[selected_i[i]], self.SMILES[selected_i[i]])
        else:

            print ("euclidean distance not implemented")

        
        # for i in X
        # for j in self.nobj
        # compute the similarity i,j
        # if d < self.param.getVal('similarity_cutoff) store j
        # alternativelly...
        # if len(results) < self.param.getVal ('similarity_num_results') store j
        # alternativelly ...
        # check if the new result is better than the workse selected, then keep, remove the worse and sort results
        
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
        return True, 'success'

    def save_space(self):
        ''' This function saves space in a pickle file '''

        # create a pickle with the names, SMILES and pre-processed reference space

        space_pkl = os.path.join(self.param.getVal('model_path'),
                                      'space.pkl')
        with open(space_pkl, 'wb') as fo:
            pickle.dump(self.nobj, fo)
            pickle.dump(self.X, fo)
            pickle.dump(self.names, fo)
            pickle.dump(self.SMILES, fo)
        return

    def load_space(self):
        ''' This function loads spacer from a pickle file '''
    
        space_pkl = os.path.join(self.param.getVal('model_path'),
                                      'space.pkl')

        with open(space_pkl, 'rb') as fo:
            self.nobj = pickle.load(fo)
            self.X = pickle.load(fo)
            self.names = pickle.load(fo)
            self.SMILES = pickle.load(fo)
        return
