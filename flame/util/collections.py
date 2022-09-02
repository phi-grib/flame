import os
import pickle
from flame.util import utils,get_logger
from datetime import datetime 
LOG = get_logger(__name__)

def createCollection(nameCollection,endpoints,versions):
    existCollection = False
    collections_list = []
    date = datetime.now().strftime('%Y-%m-%d')
    meta_path = os.path.dirname(os.path.dirname(__file__))
    collections_file = os.path.join(meta_path, 'collections.pkl')
    # check if exist file
    if os.path.isfile(collections_file):
        with open(collections_file,"rb") as file:
            collections_list = pickle.load(file)
        existCollection = checkNameCollection(collections_list,nameCollection)
        if existCollection:
            for idx, x in enumerate(collections_list):
                if x['name'] == nameCollection:
                    del collections_list[idx]
    # add new collection
    collections_list.append({'name':nameCollection,'endpoints':endpoints,'versions':versions,'date':date})
    print(collections_list)
    with open(collections_file,"wb") as file:
        pickle.dump(collections_list,file)
        LOG.info(f'Save collections.pkl file \n')

    return True,"Save collection successfully"


def deleteCollection(nameCollection):
    meta_path = os.path.dirname(os.path.dirname(__file__))
    collections_file = os.path.join(meta_path, 'collections.pkl')
    deleted = False
    listCollections = get_collections()
    for idx, x in enumerate(listCollections):
        if x['name'] == nameCollection:
            del listCollections[idx]
            deleted = True
    if deleted:
        with open(collections_file,"wb") as file:
            pickle.dump(listCollections,file)
            LOG.info(f'Save collections.pkl file \n')
            file.close()
        return deleted,"Deleted Succesfully"
    else:
        return deleted,"Failed"

def checkNameCollection(collections_list, nameCollection):
    return any(x['name'] == nameCollection for x in collections_list)


def get_collections():
    '''
    Retrieves the collection list
    '''
    meta_path = os.path.dirname(os.path.dirname(__file__))
    collections_file = os.path.join(meta_path, 'collections.pkl')
    if os.path.isfile(collections_file):
        with open(collections_file,"rb") as file:
            collections_list = pickle.load(file)
            
            return collections_list
    
    return False