#! -*- coding: utf-8 -*-

# Description    Verification process
#
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
#                Adrian Cabrera
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

from flame.documentation import Documentation
from flame.util import utils,get_logger 
import os
from rdkit import Chem,DataStructs
import pickle


LOG = get_logger(__name__)

# 1-Data cheking: Documentation
def verify_documentation (endpoint, version=None):
    '''
      Check that the required fields are completed
    '''

    blacklist = ['Species','Limits_applicability','Experimental_protocol','location','description','endpoint_positive','endpoint_negative','raw_data_url','test_set_size','training_set_url','test_set_url','bootstrap','ccp_alpha','criterion','max_depth','max_features','max_leaf_nodes','max_samples','min_impurity_decrease','min_impurity_split','min_samples_leaf','min_samples_split','min_weight_fraction_leaf','n_estimators','n_jobs','oob_score','random_state','verbose','warm_start','confidence','ACP_sampler','KNN_NN','aggregated','aggregation_function','conformal_predictors','normalizing_model','Conformal_mean_interval','Conformal_accuracy','Q2','SDEP','Comments','Other_related_models','Date_of_QMRF','Date_of_QMRF_updates','QMRF_updates','References','QMRF_same_models','Mechanistic_basis','Mechanistic_references','Supporting_information','Comment_on_the_endpoint','Endpoint_data_quality_and_variability','Descriptor_selection','Internal_validation_2','External_validation']

    if endpoint is None:
        return False, 'Empty model label'
    
    # get de model repo path
    rdir = utils.model_path(endpoint, version)
    if not os.path.isfile(os.path.join(rdir, 'model-results.pkl')):
        return False, 'Info file not found' 

    doc = Documentation(endpoint, version)

    fields =  [field for field in doc.empty_fields() if field not in blacklist]

    if fields:
        result = {'status':'Failed','comments':'fields not completed','Information':fields}
    else:
        result = {'status':'Passed','comments':'All fields required are completed','Information':[]}
    
    return True,result
    
# 1-Data cheking: data
# Manually verification
def verify_data (endpoint, version=None):
    '''TO DO'''
    return True, {'status':'Passed','comments':'','Information':['Manually verification',]}

# 1-Data cheking: prediction
def verify_prediction (endpoint, version=None):
    ''' TO DO '''
    meta_path = utils.model_path(endpoint, version)
    training_file = os.path.join(meta_path, 'training_series')
    if not os.path.isfile(training_file):
        return True, {'status':'Failed','comments':'','Information':[]}

    return True, {'status':'Passed','comments':'','Information':[]}

# 2- Model testing
def verify_model(endpoint, version= None):
      ''' TO DO'''
      doc = Documentation(endpoint, version)
      list_mols = doc.get_mols()
      api = utils.connect_api()
      count = 1
      invalid = []
      for mol in list_mols:
          toxhub_smiles = utils.getSmilesByAPI(api,mol)
          if toxhub_smiles:
              fp1,fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles(list_mols[mol])),Chem.RDKFingerprint(Chem.MolFromSmiles(toxhub_smiles))
              similarity = DataStructs.TanimotoSimilarity(fp1,fp2)
              if similarity < 0.99:
                  invalid.append({'namedrug':mol,'input_smiles':list_mols[mol],'toxhub_smiles':toxhub_smiles,'similarity':similarity}) 
          else:
              print(count,". Not found:",mol)
              count +=1

      if invalid:
          return True,{'status':'Failed','comments':'The chemical structure of the following drugs is different from that obtained in ToxHub.','Information':invalidMols}

      return True,{'status':'Passed','comments':'','Information':[]}


# 3-Inspection of Model

def inspection_model():

    return None

# 4-Examination of Executive summary

def executive_summary():
    
    return None
    

def verify (endpoint, version=None):
    result = {}
    # 1- Data cheking: Documentation
    success,  result['documentation'] = verify_documentation (endpoint, version)
    
    if not success:
        return False, result
    # 1- Data cheking: data
    success, result['data'] = verify_data (endpoint, version)

    if not success:
        return False, result
    # 1- Data cheking: prediction
    success, result['prediction'] = verify_prediction (endpoint, version)

    if not success:
        return False, result

    # save datacheking data
    datacheking = {'Data cheking':result}

    result = {}

    # 2- Model testing
    success, result['model'] = verify_model(endpoint, version)
    if not  success:
        return False, result
    
    # save model testing data
    modeltesting = {'Model testing': result}
    
    
    datacheking.update(modeltesting) # concatenates the dictionary of data cheking and the dictionary of model testing
    
    meta_path = utils.model_path(endpoint, version)
    verification_file = os.path.join(meta_path, 'verification.pkl')

    #Save in the model folder verification.pkl
    file = open(verification_file,"wb")
    pickle.dump(datacheking,file)
    file.close()
    LOG.info(f'Save verification.pkl file \n')

    # show first step of verification process
    show_result(datacheking['Data cheking'])

    return True, datacheking


def get_verification(endpoint,version):
    '''
    Retrieves the model verification if it exists
    '''
    verification = False
    meta_path = utils.model_path(endpoint, version)
    verification_file = os.path.join(meta_path, 'verification.pkl')

    if os.path.isfile(verification_file):
        file = open(verification_file,"rb")
        verification = pickle.load(file)
        file.close()
        return True,verification

    return False

#pending changes: improve scalability
#currently it is only useful for the first step of verification.
def show_result(result):
     '''
     Shows the model verification in the terminal
     '''
     if result:
         # HEADERS
         print("{:<18}{:<10}{:<40}{:<10}".format('Stage','Status','Comments','Information'),"\n")
        
         for x in result:
             information = " ".join(result[x]['Information'])
             print("{:<18}{:<10}{:<40}{:<10}".format(x,result[x]['status'],result[x]['comments'],information))
     else:
         LOG.error("Unable to print verification result")
    

    
