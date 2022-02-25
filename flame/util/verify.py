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

import pickle
from flame.documentation import Documentation
import flame.chem.sdfileutils as sdfutils
from flame.util import utils,get_logger 
import os
from rdkit import Chem,DataStructs
import yaml
import urllib3
urllib3.disable_warnings()
import requests
LOG = get_logger(__name__)

try:
    from decouple import config
except:
    LOG.error('decouple library is not installed.')
    LOG.info('pip install python-decouple')


def verify_model(endpoint, version=None):
    ''' '''
    api = connect_api()
    invalid = []
    not_found_list = []
    if not isinstance(api, requests.models.Response):
        return False,{'status':'Aborted','comments':'Failed connection to External Service'}

    try:
        doc = Documentation(endpoint, version)
    except:
        return False,{'status':'Aborted','comments':f'{endpoint} documentation.yaml not found.'}

    smiles_list = dict(zip(doc.get_names(),doc.get_smiles()))
    for drugname,smiles in smiles_list.items():
        ext_service_smiles = getSmilesByApi(api,drugname)
        if ext_service_smiles:
            fp1,fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles_list[drugname])),Chem.RDKFingerprint(Chem.MolFromSmiles(ext_service_smiles))
            similarity = DataStructs.TanimotoSimilarity(fp1,fp2)
            if similarity < 0.99:
                invalid.append(
                    {
                        'drugname': drugname,
                        'input_smiles':smiles,
                        'ext_service_smiles':ext_service_smiles,
                        'similarity':similarity,
                    })
        else:
            not_found_list.append(drugname)

    if invalid or not_found_list:
        return True,{'status':'Failed',
                     'comments':'The chemical structure of the following drugs is different from that obtained in External Service.',
                     'Information':invalid,
                     'Extra_Information':not_found_list}

    return True,{'status':'Passed'}


def getSmilesByApi(response,name):
    token = response.json()['access_token']
    # refresh_token = response.json()['refresh_token']
    headers = {'Authorization': f'Bearer {token}'}
    for _ in range(3):
        # acces to Chemistry Service
        r = requests.get("https://test.toxhub.etransafe.eu/chemistryservice.kh.svc/v1/name_to_structure",verify=False,params={'name':name}, headers=headers)

        if r.status_code == 200:
            if 'result' in r.json():
                return r.json()['result'][0]['smiles']

            print(r.json()['Empty response']+name)
            return None
        if r.status_code == 401:
            print('failed to reconnect')

def connect_api():
    
    KC_URL = config('KC_URL')
    KC_USERNAME = config('KC_USERNAME')
    PASSWORD = config('PASSWORD')
    CLIENT_SECRET = config('CLIENT_SECRET')

    #get token
    payload = f"grant_type=password&client_id=knowledge-hub&client_secret={CLIENT_SECRET}&username={KC_USERNAME}" + \
              f"&password={PASSWORD}"

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(
        f'{KC_URL}/auth/realms/KH/protocol/openid-connect/token',
        data=payload,
        headers=headers,
        verify=False,
    )
    if response.status_code != 200:
        LOG.error(response.status_code)
        return None 

    LOG.info('Succesfully connection')
    return response


def verify_SDFile_activity(endpoint,version=None):
    ''' '''
    # I check that the model label exists only in the first function of the verification process.
    # to avoid rechecking it in the following steps
    if endpoint is None:
        return False, {'status':'Aborted','comments':'Empty model label'}

    param = None
    meta_path = utils.model_path(endpoint, version)
    parameters_file_name = os.path.join(meta_path, 'parameters.yaml')
    ifile = os.path.join(meta_path, 'training_series')
    with open(parameters_file_name, 'r') as pfile:
                param = yaml.safe_load(pfile)

    # Initiate a RDKit SDFile iterator to process the molecules one by one
    suppl = Chem.SDMolSupplier(ifile,sanitize=True)
    
    # check if the activity label is defined
    if param['SDFile_activity']['value'] is  None:
        return False,{'status':'Aborted','comments':'The activity field is not specified'}

    # Iterate for every molecule inside the SDFile
    bio = None
    obj_num = 0
    list_mol_names = []
    for mol in suppl:
        if mol is None:
            LOG.error(f'(@extractInformaton) Unable to process molecule #{obj_num+1}'
                    f' in file {ifile}')
            continue

        # extract the molecule name, using a sdfileutils algorithm
        name = sdfutils.getName(
        mol, count=obj_num, field=param['SDFile_name']['value'])
        # extract biological information (activity)
        bio = sdfutils.getVal(mol, param['SDFile_activity']['value'])
        
        if bio is None:
            list_mol_names.append(name)
    
        obj_num +=1

    if list_mol_names:
        result = {'status':'Failed','comments':'The activity must be present in all molecules.','Information':list_mol_names}
    else:
        result = {'status':'Passed','comments':'','Information':list_mol_names}

    return True,result  

def verify_library(endpoint, version=None):
    ''''''
    param = None
    meta_path = utils.model_path(endpoint, version)
    parameters_file_name = os.path.join(meta_path, 'parameters.yaml')
    with open(parameters_file_name, 'r') as pfile:
                param = yaml.safe_load(pfile)

    model_pkl = os.path.join(param['model_path']['value'],'estimator.pkl')
    LOG.debug(f'Loading model from pickle file, path: {model_pkl}')
    try:
        with open(model_pkl,"rb") as input_file:
            dict_estimator = pickle.load(input_file)

    except FileNotFoundError:
        LOG.error(f'No valid model estimator found at: {model_pkl}')
        return False, {'status':'Aborted','comments':f'No valid model estimator found at: {model_pkl}'}

    # check if the pickle was created with a compatible version (currently, 1)
    if dict_estimator['version'] is not 1:
        return True, {'status':'Failed','comments':'Incompatible model version','Information':[]}

    # check if the libraries used to build this model are similar to current libraries
    if 'libraries' not in dict_estimator:
        return False, {'status':'Failed',
                       'comments':'The libraries with which the model was built have not been found in the estimator.pkl'}

    success,results = utils.compatible_modules(dict_estimator['libraries'])

    if not success:
        return True,{'status':'Failed','comments':'Incompatible libraries have been found','Information':results}
    else:
         return True,{'status':'Passed'}


def verify_documentation (endpoint, version=None):
    '''
      Check that the required fields are completed
    '''
    blacklist = ['Species','Limits_applicability','Experimental_protocol','location','description','endpoint_positive','endpoint_negative','raw_data_url',
    'test_set_size','training_set_url','test_set_url','bootstrap','ccp_alpha','criterion','max_depth','max_features','max_leaf_nodes','max_samples',
    'min_impurity_decrease','min_impurity_split','min_samples_leaf','min_samples_split','min_weight_fraction_leaf','n_estimators','n_jobs','oob_score',
    'random_state','verbose','warm_start','confidence','ACP_sampler','KNN_NN','aggregated','aggregation_function','conformal_predictors','normalizing_model',
    'Conformal_mean_interval','Conformal_accuracy','Q2','SDEP','Comments','Other_related_models','Date_of_QMRF','Date_of_QMRF_updates','QMRF_updates',
    'References','QMRF_same_models','Mechanistic_basis','Mechanistic_references','Supporting_information','Comment_on_the_endpoint','Endpoint_data_quality_and_variability',
    'Descriptor_selection','Internal_validation_2','External_validation']

    doc = Documentation(endpoint, version)
    fields =  [field for field in doc.empty_fields() if field not in blacklist]

    if fields:
        result = {'status':'Failed','comments':'Missing required information.','Information':fields}
    else:
        result = {'status':'Passed','comments':'All fields required are completed.','Information':[]}
    
    return True,result

def verify (endpoint, version=None):
    result = {}

    # 1.0 Data checking: activity
    success, result['activity'] = verify_SDFile_activity(endpoint, version)
    if not success:
        return False,result
    
    # 1.1 Data cheking: Check the validity of the structure provided
    success, result['model'] = verify_model(endpoint, version)
    if not success:     
        return False,result
    
    # save data checking step
    datachecking = {'Data checking':result}
    
    # 2.0 Model testing: Check library
    result = {}
    success,result['libraries'] = verify_library(endpoint,version)
    if not success:
        return False,result

    # save model testing step
    modeltesting = {'Model testing':result}

    datachecking.update(modeltesting)

    # 3- Documentation: required fields
    result = {}
    success,  result['fields'] = verify_documentation (endpoint, version)
    if not success:
        return False, result
    
    # save documentation step
    documentation = {'Documentation': result}

    datachecking.update(documentation) # concatenates the 3 steps
    meta_path = utils.model_path(endpoint, version)
    verification_path = os.path.join(meta_path, 'verification.yaml')
    
    #Save in the model folder verification.yaml
    with open(verification_path,'w') as file:
        yaml.dump(datachecking,file)
    
    
    # show first step of verification process
    # show_result(datachecking['Data checking'])

    return True, datachecking


def get_verification(endpoint,version):
    '''
    Retrieves the model verification if it exists
    '''
    verification = False
    meta_path = utils.model_path(endpoint, version)
    verification_path = os.path.join(meta_path, 'verification.yaml')

    if os.path.isfile(verification_path):
        with open(verification_path,'r') as file:
            verification = yaml.load(file,Loader=yaml.FullLoader)
            
        return True,verification

    return False

#pending changes: improve scalability
#currently it is only useful for the first step of verification.
# def show_result(result):
#      '''
#      Shows the model verification in the terminal
#      '''
#      if result:
#          # HEADERS
#          print("{:<18}{:<10}{:<40}{:<10}".format('Stage','Status','Comments','Information'),"\n")
        
#          for x in result:
#              information = " ".join(result[x]['Information'])
#              print("{:<18}{:<10}{:<40}{:<10}".format(x,result[x]['status'],result[x]['comments'],information))
#      else:
#          LOG.error("Unable to print verification result")
    

    
