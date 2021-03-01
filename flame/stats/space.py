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
# from rdkit.Chem import AllChem
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

        self.nobj, self.nvarx = np.shape(self.X)

        self.Xref = None # metric of reference spaced, loaded in searches only 
        self.Dmax = 1000.0 # an arbitrary value

    def _buildMorganFP (self):
        t1 = time.time()
        Xbit = [DataStructs.cDataStructs.CreateFromBitString("".join(i.astype(str))) for i in self.X]
        LOG.info (f'{self.nobj} fingerprints converted in time: {time.time()-t1:.4f} secs')
        self.X = Xbit
        self.Dmax = 1.0
        return

    def _buildSubStructure(self):
        t1 = time.time()
        Xbit = [DataStructs.cDataStructs.CreateFromBitString("".join(i.astype(str))) for i in self.X]
        LOG.info (f'{self.nobj} fingerprints converted in time: {time.time()-t1:.4f} secs')
        self.X = Xbit
        self.Dmax = 1.0
        return

    def _buildMD (self):
        ydist = distance.pdist(self.X, metric='Euclidean')
        #print ('min:', np.min(ydist), 'max:', np.max(ydist))
        self.Dmax = np.percentile(ydist,95)
        return

    def build(self):
        ''' This function pre-process the X matrix, optimizing it for searching in the case
            of fingerprints 
        '''
        results = []
        results.append(('nobj', 'number of objects', self.nobj))

        MDs = self.param.getVal('computeMD_method')
        if   'morganFP' in MDs:
            self._buildMorganFP ()
        elif 'substructureFP' in MDs:
            self._buildSubStructure ()
        else:
            self._buildMD ()

        results.append(('dmax', 'perecentil 95 of internal distances', self.Dmax))

        return True, results

    def _searchMorganFP (self, cutoff, numsel, metric):

        incompatible = ['Substructural']
        if metric is None:
            metric = 'Tanimoto'
        elif metric in incompatible:
            LOG.warning (f'Metric {metric} is not compatible with the descriptors present in this space')
            metric = 'Tanimoto'

        results = []

        t1 = time.time()

        # for each compound in the search set 
        for ivector in self.X:
            bitestring="".join(ivector.astype(str))
            ifp = DataStructs.cDataStructs.CreateFromBitString(bitestring)

            selected_i = []
            selected_d = []
            
            d_worst = 0.000

            #TODO Check speed BulkTanimoto
            # for each compound in the space
            for j, jvector in enumerate(self.Xref):

                d = DataStructs.FingerprintSimilarity(ifp,jvector, metric=DataStructs.TanimotoSimilarity)

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
            for oi in self.objinforef:
                results_info[oi] = []        # all the objects information (name, smiles, ID, activity, etc.)

            for sd,si in zip(selected_d, selected_i):
                results_info['distances'].append(sd)
                for oi in self.objinforef:
                    results_info[oi].append(self.objinforef[oi][si])

            results.append(results_info)

        LOG.info (f'search completed in time: {time.time()-t1:.4f} secs')

        return True, results

    def _searchSubStructure (self, numsel, metric):

        if metric is None:
            metric = 'Substructural'
        elif metric != 'Substructural':
            LOG.warning (f'Metric {metric} is not compatible with the descriptors present in this space. Using Substructural')
            metric = 'Substructural'

        results = []

        t1 = time.time()

        # for each compound in the search set 
        for i, ivector in enumerate(self.X):
            bitestring="".join(ivector.astype(str))
            ifp = DataStructs.cDataStructs.CreateFromBitString(bitestring)

            nselected = 0
            selected_i = []
            selected_d = []
            
            # for each compound in the space
            for j, jfp in enumerate(self.Xref):
                if DataStructs.AllProbeBitsMatch(ifp, jfp):
                    mi = Chem.MolFromSmiles(self.objinfo['SMILES'][i])
                    mj = Chem.MolFromSmiles(self.objinforef['SMILES'][j])

                    if mj.HasSubstructMatch(mi):
                        selected_i.append(j)
                        selected_d.append(1.00)
                        nselected+=1

                if nselected > numsel:
                    break

            # results for molecule i are stored in a dictionary
            results_info = {}
            results_info['distances'] = []   # distances are allways stored
            for oi in self.objinforef:
                results_info[oi] = []        # all the objects information (name, smiles, ID, activity, etc.)

            for sd,si in zip(selected_d, selected_i):
                results_info['distances'].append(sd)
                for oi in self.objinforef:
                    results_info[oi].append(self.objinforef[oi][si])

            results.append(results_info)

        LOG.info (f'search completed in time: {time.time()-t1:.4f} secs')

        return True, results

    def _searchMD (self, cutoff, numsel, metric):
        
        incompatible = ['Tanimoto', 'Substructural']
        if metric is None:
            metric = 'Euclidean'
        elif metric in incompatible:
            LOG.warning (f'Metric {metric} is not compatible with the descriptors present in this space')
            metric = 'Euclidean'

        results = []

        # for each compound in the search set 
        for ivector in self.X:

            selected_i = []
            selected_d = []
            
            d_worst = 0.000

            # for each compound in the space
            for j, jvector in enumerate(self.Xref):

                if metric == 'Euclidean':
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
            for oi in self.objinforef:
                results_info[oi] = []        # all the objects information (name, smiles, ID, activity, etc.)

            for sd,si in zip(selected_d, selected_i):
                results_info['distances'].append(sd)
                for oi in self.objinforef:
                    results_info[oi].append(self.objinforef[oi][si])

            results.append(results_info)

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

        MDs = self.param.getVal('computeMD_method')
        if   'morganFP' in MDs:
            return self._searchMorganFP (cutoff, numsel, metric)
        elif 'substructureFP' in MDs:
            return self._searchSubStructure (numsel, metric)
        else:
            return self._searchMD (cutoff, numsel, metric)

        return False, 'unexpected condition'


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
            self.objinforef = pickle.load(fo)
        return
