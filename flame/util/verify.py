from flame.documentation import Documentation
from flame.util import utils
import os


def verify_documentation (endpoint, version=None):

    if endpoint is None:
        return False, 'Empty model label'
    
    # get de model repo path
    rdir = utils.model_path(endpoint, version)
    if not os.path.isfile(os.path.join(rdir, 'model-results.pkl')):
        return False, 'Info file not found' 

    doc = Documentation(endpoint, version)

    # implement empty_fields method in documentation that iterates the fields
    # and returns a list with those that are empty
    return True, doc.empty_fields()

def verify_data (endpoint, version=None):
    print ('verify_data', endpoint, version)
    return True, 'OK'

def verify_prediction (endpoint, version=None):
    print ('verify_prediction', endpoint, version)
    return True, 'OK'

def verify (endpoint, version=None):
    
    result = {}
    success,  result['documentation'] = verify_documentation (endpoint, version)
    if not success:
        return False, result

    success, result['data'] = verify_data (endpoint, version)
    if not success:
        return False, result

    success, result['prediction'] = verify_prediction (endpoint, version)
    if not success:
        return False, result

    print (result)
    
    return True, result
