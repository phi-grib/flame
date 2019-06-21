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
        
        if numsel is None:
            numsel = len(self.X)

        if self.param.getVal('computeMD_method') == ['morganFP']:
            
            # for each compound in the search set 
            for i, inpvector in enumerate(X):
                bitestring="".join(inpvector.astype(str))
                ivector = DataStructs.cDataStructs.CreateFromBitString(bitestring)

                # for each compound in the space
                selected_i = []
                selected_d = []
                for j, jvector in enumerate(self.X):
                    d = DataStructs.FingerprintSimilarity(ivector,jvector, metric=DataStructs.TanimotoSimilarity)
                    
                    if d <= cutoff:
                        continue

                    # if results set is not completed add
                    if len(selected_i) < numsel:
                        selected_i.append(j)
                        selected_d.append(d)
                        z = sorted (zip(selected_d,selected_i),reverse=True)
                        selected_d = [x for x,_ in z]
                        selected_i = [x for _,x in z]

                    # otherwyse, compare the new d with the min d
                    else:
                        if d > selected_d[-1]:   # better than worse compound                           
                            #add at the beggining 
                            selected_i[-1]=j
                            selected_d[-1]=d
                            z = sorted (zip(selected_d,selected_i),reverse=True)
                            selected_d = [x for x,_ in z]
                            selected_i = [x for _,x in z]

                for sd,si in zip(selected_d, selected_i):
                    print (i, sd, self.names[si], self.SMILES[si])
        else:
            # TODO: implement euclidean
            print ("euclidean distance not implemented")

        # TODO: return results
        # results = {
        #     'matrix': xmatrix,
        #     'names': md_name,
        #     'success_arr': success_list
        # }

        return True, "ok"

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
