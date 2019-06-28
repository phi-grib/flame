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
        '''Initializes the chemical space'''
        self.param = param
        self.Dmax = 1000.0 # an arbitrary value

    def build(self, X, names, SMILES):
        ''' This function pre-process the X matrix, optimizing it for searching in the case
            of fingerprints 
        '''

        self.names = names
        self.SMILES = SMILES
        self.nobj, self.nvarx = np.shape(X)

        if len (self.param.getVal('computeMD_method')) > 1:
            return False, 'Only a single type of MD can be used to compute similarity'

        # if X contains fingerprints as numpy, convert to RDKit BitVector to speed-up
        # future similarity searches
        if self.param.getVal('computeMD_method')[0] in ['morganFP']: # include any RDKit fingerprint here
            self.X = []
            for i in X:
                self.X.append(DataStructs.cDataStructs.CreateFromBitString("".join(i.astype(str))))
            self.Dmax = 1.0
        else:
            ydist = distance.pdist(X, metric='euclidean')
            #print ('min:', np.min(ydist), 'max:', np.max(ydist))
            self.Dmax = np.percentile(ydist,95)
            self.X = X

        results = []
        results.append(('nobj', 'number of objects', self.nobj))

        if self.Dmax is not 1.0:
            results.append(('dmax', 'perecentil 95 of internal distances', self.Dmax))

        return True, results


    def search (self, X, cutoff, numsel, metric):
        ''' This function searches for compounds in the chemical space similar to the compounds of input file
            already characterized by the X matrix

            the metric and the cutoff used for the search (distance cutoff and number to extract) are
            defined as parameters
        '''

        #print ('start')
        # load pickle with reference space
        self.load_space()
        #print ('pickle loaded')

        if cutoff is None:
            cutoff = 0.0
        
        if numsel is None:
            numsel = len(self.X)

        results = []
        
        # for fingerprint MD
        isFingerprint = (self.param.getVal('computeMD_method') == ['morganFP'])

        # for each compound in the search set 
        for i, ivector in enumerate(X):

            if isFingerprint:
                bitestring="".join(ivector.astype(str))
                ifp = DataStructs.cDataStructs.CreateFromBitString(bitestring)

            # for each compound in the space
            selected_i = []
            selected_d = []
            #print ('searching compound:', i)
            
            d_worst = 0.000

            for j, jvector in enumerate(self.X):

                
                if metric == 'Tanimoto':
                    d = DataStructs.FingerprintSimilarity(ifp,jvector, metric=DataStructs.TanimotoSimilarity)
                elif metric == 'Euclidean':
                    d = 1.000-(distance.euclidean(ivector,jvector)/self.Dmax)

                if d <= cutoff:
                    continue

                # if results set is not completed add
                if len(selected_i) < numsel:
                    selected_i.append(j)
                    selected_d.append(d)
                    z = sorted (zip(selected_d,selected_i),reverse=True)
                    selected_d = [x for x,_ in z]
                    selected_i = [x for _,x in z]

                    d_worst = selected_d[-1]

                    # if the worst compound is identical, we cannot improve the search 
                    if d_worst == 1.000:
                        break

                # otherwyse, compare the new d with the min d
                else:
                    if d > d_worst:   # better than worse compound                           
                        #replace worst
                        selected_i[-1]=j
                        selected_d[-1]=d
                        z = sorted (zip(selected_d,selected_i),reverse=True)
                        selected_d = [x for x,_ in z]
                        selected_i = [x for _,x in z]
    
                        d_worst = selected_d[-1]

                        # if the worst compound is identical, we cannot improve the search 
                        if d_worst == 1.000:
                            break


            #print ('completed')
            results_distances = []
            results_names = []
            results_smiles = []

            for sd,si in zip(selected_d, selected_i):
                results_distances.append(sd)
                results_names.append(self.names[si])
                results_smiles.append(self.SMILES[si])
                
                #print (i, sd, self.names[si], self.SMILES[si])

            results.append({'distances':results_distances,
                            'names':results_names,
                            'SMILES':results_smiles
            })

    
        return True, results


    def save_space(self):
        ''' This function saves the chemical space in a pickle file '''

        space_pkl = os.path.join(self.param.getVal('model_path'),
                                      'space.pkl')
        with open(space_pkl, 'wb') as fo:
            pickle.dump(self.nobj, fo)
            pickle.dump(self.X, fo)
            pickle.dump(self.names, fo)
            pickle.dump(self.SMILES, fo)
            pickle.dump(self.Dmax, fo)
        return


    def load_space(self):
        ''' This function loads the chemical space from a pickle file '''
    
        space_pkl = os.path.join(self.param.getVal('model_path'),
                                      'space.pkl')

        with open(space_pkl, 'rb') as fo:
            self.nobj = pickle.load(fo)
            self.X = pickle.load(fo)
            self.names = pickle.load(fo)
            self.SMILES = pickle.load(fo)
            self.Dmax = pickle.load(fo)
        return
