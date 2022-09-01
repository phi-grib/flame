import os
import pickle
from flame.util import utils,get_logger
from datetime import datetime 
LOG = get_logger(__name__)

def createCollection(nameCollection,endpoints,versions):
    collections_list = []
    date = datetime.now().strftime('%Y-%m-%d')
    meta_path = os.path.dirname(os.path.dirname(__file__))
    collections_file = os.path.join(meta_path, 'collections.pkl')
    # check if exist collection
    if os.path.isfile(collections_file):
        with open(collections_file,"rb") as file:
            collections_list = pickle.load(file)
            
    # add new collection
    collections_list.append({'name':nameCollection,'endpoints':endpoints,'versions':versions,'date':date})
    print(collections_list)
    with open(collections_file,"wb") as file:
        pickle.dump(collections_list,file)
        LOG.info(f'Save collections.pkl file \n')

    return "Save collection successfully"


def get_collections():
    '''
    Retrieves the collection list
    '''
    meta_path = os.path.dirname(os.path.dirname(__file__))
    collections_file = os.path.join(meta_path, 'collections.pkl')
    if os.path.isfile(collections_file):
        with open(collections_file,"rb") as file:
            collections_list = pickle.load(file)
            file.close()
            return collections_list
    
    return False