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
import time

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from flame.util import utils, get_logger

LOG = get_logger(__name__)


class Space:
    def __init__(self, param, conveyor):
        '''Initializes the chemical space'''
        
        self.param = param
        self.conveyor = conveyor

        self.objinfo = {}
        itemlist = ['obj_nam', 'obj_id', 'SMILES', 'ymatrix']
        for item in itemlist:
            item_val = self.conveyor.getVal(item)
            if item_val is not None:
                self.objinfo[item] = item_val

        MDs = self.param.getVal('computeMD_method')
        self.isFingerprint = ('morganFP' in MDs)

        self.X = None

        if self.isFingerprint and len(MDs)>1:
            fingerprint_index = self.conveyor.getVal('fingerprint_index')
        
            if fingerprint_index is None:
                return False, 'Only a single type of MD can be used to compute similarity'
            else:
                LOG.warning("Flame cannot combine fingerprints and continuous variables for computing similarity. Only the fingerprints will be used for showing similar compounds.")
                self.X = self.conveyor.getVal('xmatrix')[:,fingerprint_index]
        else:
            self.X = self.conveyor.getVal('xmatrix')

        self.Xref = None # metric of reference spaced, loaded in searches only 
        self.Dmax = 1000.0 # an arbitrary value

    def build(self):
    # def build(self, X, names, ids, SMILES):
        ''' This function pre-process the X matrix, optimizing it for searching in the case
            of fingerprints 
        '''
        self.nobj, self.nvarx = np.shape(self.X)

        # if X contains fingerprints as numpy, convert to RDKit BitVector to speed-up
        # future similarity searches
        if self.isFingerprint: # include any RDKit fingerprint here
            t1 = time.time()
            Xbit = [DataStructs.cDataStructs.CreateFromBitString("".join(i.astype(str))) for i in self.X]
            self.X = Xbit
            print (f'time: {time.time()-t1}')

            self.Dmax = 1.0
        else:
            ydist = distance.pdist(self.X, metric='Euclidean')
            #print ('min:', np.min(ydist), 'max:', np.max(ydist))
            self.Dmax = np.percentile(ydist,95)

        results = []
        results.append(('nobj', 'number of objects', self.nobj))

        if self.Dmax is not 1.0:
            results.append(('dmax', 'perecentil 95 of internal distances', self.Dmax))

        return True, results


    def search (self, cutoff, numsel, metric):
        ''' This function searches for compounds in the chemical space similar to the compounds of input file
            already characterized by the X matrix

            the metric and the cutoff used for the search (distance cutoff and number to extract) are
            defined as parameters
        '''

        # load pickle with reference space
        self.load_space()

        # set defaults
        if cutoff is None:
            cutoff = 0.0
        
        if numsel is None:
            #numsel = len(self.X)
            numsel = 10

        # float variables only can be compared using euclidean
        if self.isFingerprint is False:
            metric = 'Euclidean'
        else:
            if metric is None:
                if self.isFingerprint :
                    metric = 'Tanimoto'
                else:
                    metric = 'Euclidean'

        results = []

        # for each compound in the search set 
        for ivector in self.X:

            if self.isFingerprint:
                bitestring="".join(ivector.astype(str))
                ifp = DataStructs.cDataStructs.CreateFromBitString(bitestring)

            # for each compound in the space
            selected_i = []
            selected_d = []
            #print ('searching compound:', i)
            
            d_worst = 0.000

            for j, jvector in enumerate(self.Xref):

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

            # results for molecule i are stored in a dictionary
            results_info = {}
            results_info['distances'] = []   # distances are allways stored
            for oi in self.objinfo:
                results_info[oi] = []        # all the objects information (name, smiles, ID, activity, etc.)

            for sd,si in zip(selected_d, selected_i):
                results_info['distances'].append(sd)
                for oi in self.objinfo:
                    results_info[oi].append(self.objinfo[oi][si])

            results.append(results_info)
    
        return True, results


    def save_space(self):
        ''' This function saves the chemical space in a pickle file '''

        space_pkl = os.path.join(self.param.getVal('model_path'),
                                      'space.pkl')
        with open(space_pkl, 'wb') as fo:
            pickle.dump(self.nobj, fo)
            pickle.dump(self.X, fo)  #the reference space matrix is self X 
            pickle.dump(self.Dmax, fo)
            pickle.dump(self.objinfo, fo)
        return


    def load_space(self):
        ''' This function loads the chemical space from a pickle file '''
    
        space_pkl = os.path.join(self.param.getVal('model_path'),
                                      'space.pkl')

        with open(space_pkl, 'rb') as fo:
            self.nobj = pickle.load(fo)
            self.Xref = pickle.load(fo) 
            self.Dmax = pickle.load(fo)
            self.objinfo = pickle.load(fo)
        return
