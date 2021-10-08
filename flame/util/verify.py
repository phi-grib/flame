from flame.documentation import Documentation
from flame.util import utils
import os
from flame.conveyor import Conveyor
from rdkit import Chem,DataStructs


def verify_documentation (endpoint, version=None):
    blacklist = ['Species','Limits_applicability','Experimental_protocol','location','description','endpoint_positive','endpoint_negative','raw_data_url','test_set_size','training_set_url','test_set_url','bootstrap','ccp_alpha','criterion','max_depth','max_features','max_leaf_nodes','max_samples','min_impurity_decrease','min_impurity_split','min_samples_leaf','min_samples_split','min_weight_fraction_leaf','n_estimators','n_jobs','oob_score','random_state','verbose','warm_start','confidence','ACP_sampler','KNN_NN','aggregated','aggregation_function','conformal_predictors','normalizing_model','Conformal_mean_interval','Conformal_accuracy','Q2','SDEP','Comments','Other_related_models','Date_of_QMRF','Date_of_QMRF_updates','QMRF_updates','References','QMRF_same_models','Mechanistic_basis','Mechanistic_references','Supporting_information','Comment_on_the_endpoint','Endpoint_data_quality_and_variability','Descriptor_selection','Internal_validation_2','External_validation']
    if endpoint is None:
        return False, 'Empty model label'
    
    # get de model repo path
    rdir = utils.model_path(endpoint, version)
    if not os.path.isfile(os.path.join(rdir, 'model-results.pkl')):
        return False, 'Info file not found' 

    doc = Documentation(endpoint, version)

    fields =  [field for field in doc.empty_fields() if field not in blacklist]

    if(fields):
        result = {'status':'Failed','comments':'fields not completed','Information':fields}
    else:
        result = {'status':'Passed','comments':'All fields required are completed','Information':''}

    return True,result


# Manually verification
def verify_data (endpoint, version=None):
    print ('verify_data', endpoint, version)
    return True, {'status':'Passed','comments':'','Information':'Manually verification'}



def verify_prediction (endpoint, version=None):
    print ('verify_prediction', endpoint, version)

    
    meta_path = utils.model_path(endpoint, version)
    training_file = os.path.join(meta_path, 'training_series')

    if not os.path.isfile(training_file):
        return True, {'status':'Failed','comments':'','Information':'TO DO'}

    return True, {'status':'Passed','comments':'','Information':'TO DO'}


 # TO DO
def verify_model(endpoint, version= None):
    doc = Documentation(endpoint, version)
    list_of_mols = doc.get_mols()
    # rdkit process to verify similitud 
    list_of_mols = {'furosemide':'NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl'}
    fake_api = {'furosemide':'NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl'}
    invalidMols = {}

    for mol in list_of_mols:
        #apiSmile recuperaria el smile desde la api
        apiSmile = fake_api[mol]
        if apiSmile:

            localSmile,apiSmile = Chem.MolFromSmiles(list_of_mols[mol]),Chem.MolFromSmiles(apiSmile)
            fp1,fp2 = Chem.RDKFingerprint(localSmile),Chem.RDKFingerprint(apiSmile)

            if DataStructs.TanimotoSimilarity(fp1,fp2) < 0.99:
                invalidMols[mol] = localSmile    
        
        print(invalidMols)

    

    return True,{'status':'Passed','comments':'','Information':'TO DO'}

def verify (endpoint, version=None):
    
    result = {}
    success,  result['documentation'] = verify_documentation (endpoint, version)
    success, result['model'] = verify_model(endpoint, version)
    print(result['model'])

    if not success:
        return False, result

    success, result['data'] = verify_data (endpoint, version)
    if not success:
        return False, result

    success, result['prediction'] = verify_prediction (endpoint, version)
    if not success:
        return False, result

    return True, result
