import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from scipy.spatial import distance 
import pickle

#xmatrix = np.array(fp1, dtype = 'bool')

xmatrix = np.empty((2, 2048), dtype=np.int8)

mol = Chem.MolFromSmiles('[H][C@]12[C@H](C[C@@H](C)C=C1C=C[C@H](C)[C@@H]2CC[C@@H]1C[C@@H](O)CC(=O)O1)OC(=O)C(C)(C)CC')
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
DataStructs.ConvertToNumpyArray(fp1,xmatrix[0])

mol = Chem.MolFromSmiles('CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(=C(N1CC[C@@H](O)C[C@@H](O)CC(O)=O)C1=CC=C(F)C=C1)C1=CC=CC=C1')
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
DataStructs.ConvertToNumpyArray(fp2,xmatrix[1])


with open('nfingers.pkl', 'wb') as fo:
    pickle.dump(xmatrix, fo)

with open('rfingers.pkl', 'wb') as fo:
    pickle.dump(fp1, fo)
    pickle.dump(fp2, fo)


# xmatrix = np.vstack((xmatrix, fp2))

print ('start')
for i in range (1000000):
    d = DataStructs.FingerprintSimilarity(fp1,fp2, metric=DataStructs.TanimotoSimilarity)
print (d)

# d = DataStructs.FingerprintSimilarity(xmatrix[0],xmatrix[1], metric=DataStructs.TanimotoSimilarity)

print ('start')
x1 = xmatrix[0]
x2 = xmatrix[1]
for i in range (100):
    d = 1.0-distance.jaccard(x1, x2)
print (d)

# fp1 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol1, 8), dtype='bool')
# fp2 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol2, 8), dtype='bool')

def tanimoto(v1, v2):
    """
    Calculates tanimoto similarity for two bit vectors*
    """
    return(np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum())

v1 = xmatrix[0]
v2 = xmatrix[1]

print ('start')
for i in range (100):
    #d = tanimoto(x1, x2)
    d = (np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum())

print (d)

# print (xmatrix[0])
# m1 = DataStructs.cDataStructs.CreateFromBinaryText(xmatrix[0])
# m2 = DataStructs.cDataStructs.CreateFromBinaryText(xmatrix[1])

# d = DataStructs.FingerprintSimilarity(m1,m2, metric=DataStructs.TanimotoSimilarity)
# print (d)

b1="".join(xmatrix[0].astype(str))
b2="".join(xmatrix[1].astype(str))

n1 = DataStructs.cDataStructs.CreateFromBitString(b1)
n2 = DataStructs.cDataStructs.CreateFromBitString(b2)

d = DataStructs.FingerprintSimilarity(n1,n2, metric=DataStructs.TanimotoSimilarity)

print (d)




# arr = np.zeros((0,), dtype=np.int32)
# print (arr)
# DataStructs.ConvertToNumpyArray(fp,arr)
# print (arr)