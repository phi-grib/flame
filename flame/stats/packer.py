import os
import pickle
import numpy as np


def load_estimators (estimator, transform):
    estimator_path = './estimator.pkl')
    with open(estimator_path, 'rb') as f:
        estimator_dict = pickle.load(f)

    estimator = estimator_dict['estimador']
    transform = estimator_dict['transformador']

def meta_estimator (X):

    X = np.clip(X, -3.3, 0.3)

    # compute y using the noisy x
    xp = transform.transform(X)

    # predict returns an np.array with a single val
    return True, estimator.predict(xp)[0]


def main():
    
    estimator = None
    transform = None

    load_estimators (estimator, transform)
        
    meta_estimator_path = './meta_estimator.pkl')
    with open(meta_estimator_path, 'wb') as f:
        pickle.dump(meta_estimator, f)


if __name__ == "__main__":
    main()
