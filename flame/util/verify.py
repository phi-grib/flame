from flame.documentation import Documentation
from flame.util import utils,get_logger 
import os
from flame.conveyor import Conveyor
from rdkit import Chem,DataStructs
import pickle


LOG = get_logger(__name__)

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


# Manually verification
# TO DO
def verify_data (endpoint, version=None):
    return True, {'status':'Passed','comments':'','Information':['Manually verification',]}


# TO DO
def verify_prediction (endpoint, version=None):
    
    meta_path = utils.model_path(endpoint, version)
    training_file = os.path.join(meta_path, 'training_series')
    if not os.path.isfile(training_file):
        return True, {'status':'Failed','comments':'','Information':[]}

    return True, {'status':'Passed','comments':'','Information':[]}


# def verify_model(endpoint, version= None):
#     doc = Documentation(endpoint, version)
#     list_of_mols = doc.get_mols() #lista de mols
#     print("Total: ",len(list_of_mols))
#     api = utils.connect_api()
#     count = 1
#     countInvalidMols = 1

#     invalidMols = {}
#     for mol in list_of_mols:
#         apiSmile = utils.getSmilesByAPI(api,mol)
#         aux_smile = apiSmile
#         if apiSmile:
#             localSmile,apiSmile = Chem.MolFromSmiles(list_of_mols[mol]),Chem.MolFromSmiles(apiSmile)
#             fp1,fp2 = Chem.RDKFingerprint(localSmile),Chem.RDKFingerprint(apiSmile)

#             if DataStructs.TanimotoSimilarity(fp1,fp2) < 0.99:
#                 invalidMols[mol] = [list_of_mols[mol],aux_smile]
#                 countInvalidMols += 1
#         else:
#             print(count,". Not found:",mol)
#             count +=1

#     print("Similarity below 0.99: ",countInvalidMols)         
#     return True,{'status':'Passed','comments':'','Information':invalidMols}

def verify (endpoint, version=None):
    
    result = {}
    success,  result['documentation'] = verify_documentation (endpoint, version)
    #success, result['model'] = verify_model(endpoint, version)

    if not success:
        return False, result

    success, result['data'] = verify_data (endpoint, version)

    if not success:
        return False, result

    success, result['prediction'] = verify_prediction (endpoint, version)

    if not success:
        return False, result
    


    meta_path = utils.model_path(endpoint, version)
    verification_file = os.path.join(meta_path, 'verification.pkl')

    #Save in the model folder verification.pkl
    file = open(verification_file,"wb")
    pickle.dump(result,file)
    file.close()
    LOG.info(f'Save verification.pkl file \n')

    show_result(result)

    return True, result


def get_verification(endpoint,version):
    '''
    Retrieves the model verification if it exists
    '''
    result_verification = False
    meta_path = utils.model_path(endpoint, version)
    verification_file = os.path.join(meta_path, 'verification.pkl')

    if os.path.isfile(verification_file):
        file = open(verification_file,"rb")
        result_verification = pickle.load(file)
        file.close()
        return True,result_verification

    return False



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
    

    
