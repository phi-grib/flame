# -*- coding: utf-8 -*-

##    Description    PCA toolkit using NIPALS algorithm
##                   
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu) 
##
##    Copyright 2013 Manuel Pastor
##
##    This file is part of eTOXlab.
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scale import center, scale

class pca:

    def __init__(self):
        self.X = None
        self.A = 0 # model dimensionality
        self.nobj = 0
        self.nvar = 0
    
        self.mu = None
        self.wg = None

        self.autoscale = False

        self.SSX = 0.0
        
        self.t = []  # scores
        self.p = []  # loadings
        self.SSXex = []   # SSX explained
        self.SSXac = []   # SSX accumulated

    def saveModel(self,filename):
        """Saves the whole model to a binary file in numpy .npy format

        """

        f = file(filename,'wb')

        np.save(f,self.A)
        np.save(f,self.nobj)
        np.save(f,self.nvar)
        
        np.save(f,self.mu)
        np.save(f,self.wg)

        np.save(f,self.autoscale)

        np.save(f,self.SSX)
        
        for a in range(self.A):
            np.save(f,self.t[a])
            np.save(f,self.p[a])
            np.save(f,self.SSXex[a])
            np.save(f,self.SSXac[a])

        f.close()

    def loadModel(self,filename):
        """Loads the whole model from a binary file in numpy .npy format

        """

        f = file(filename,'rb')
        
        self.A = np.load(f)
        self.nobj = np.load(f)
        self.nvar = np.load(f)
        
        self.mu = np.load(f)
        self.wg = np.load(f)

        self.autoscale = np.load(f)

        self.SSX = np.load(f)

        for a in range(self.A):
            self.t.append (np.load(f))
            self.p.append (np.load(f))
            self.SSXex.append (np.load(f))
            self.SSXac.append (np.load(f))

        f.close()

    def build (self, X, targetA, autoscale=False):

        nobj, nvar= np.shape(X)

        self.nobj = nobj
        self.nvar = nvar

        self.X = X

        X, mu = center(X)
        X, wg = scale (X, autoscale)

        self.mu = mu
        self.wg = wg
        self.autoscale = autoscale

        SSXac=0.0

        for a in range(targetA):
            # extracts LV
            t, p = self.extractPC(X)

            self.t.append(t)
            self.p.append(p)

            # deflates X
            X, SSX, SSXex = self.deflatePC(X,t,p)

            SSXac += SSXex
            
            self.SSXex.append(SSXex)
            self.SSXac.append(SSXac)
            
            if a==0:
                self.SSX = SSX

        self.A = targetA


    def extractPC (self, X):
        """Computes a single PC from the numpy X matrix (X) provided as argument
           using the NIPALS algorithm

           NIPALS-PCA is iterative and runs until convergence. Criteria used here are:
           - less than 100 iterarios
           - changes in any p value <= 1.0E-9

           Returns two numpy vectors
           t:    scores
           p:    loadings
        """

        nobj,nvar = np.shape (X)
        p = np.zeros(nvar, dtype=np.float64)
        pold = np.zeros(nvar, dtype=np.float64)
        t = np.zeros(nobj, dtype=np.float64)
        
        ttmax = 0.00
        for k in range(nvar):
            obj = X[:,k]
            tt = np.dot(obj.T,obj)
            if tt>ttmax:
                ttmax = tt
                tti = k

        t=np.copy(X[:,tti])
     
        for iter in range (100):  # max 100 iterations

            # (ii) p' = t'X/t't
            for k in range(nvar):
                p[k] = np.dot(t.T,X[:,k])/ np.dot(t.T,t)

            # (iii) normalice P to length 1
            p /= np.sqrt(np.dot(p.T,p))

            # (iv) t = Xp/p'p)
            for j in range(nobj):
                t[j] = np.dot(X[j,:],p) / np.dot(p.T,p)

            # check convergence
            if max(pold-p) > 1.0e-9:  # convergence criteria set to 1.0e-9
                pold = np.copy(p)
            else:
                break

        return t,p


    def deflatePC (self, X, t, p):
        """Deflates the numpy X matrix (X) using the numpy scores (t) and loadings (p) 
           using the NIPALS algorithm

           Returns
           X:     the deflated X matrix
           SSX:   Sum-of-squares of the X matrix before the deflation
           SSXex: Sum-of-squares of the scores vector, hence explained by this PC
        """
        nobj,nvar = np.shape (X)
        SSX=0.0
        for i in range (nobj):
            obj = X[i,:]
            SSX += np.dot(obj.T,obj)
            obj -= (t[i]*p)
        SSXex = np.dot(t.T,t)
        return X, SSX, SSXex

    
    def projectPC (self, X, a):
        """The numpy X matrix (X) is projected into an existing PCA model to extract a single PC

           This call is repeated A times (one for each model dimension) passing the deflated X matrix in
           each call
          
           The value of a is only used to check if this is the first call. If true, the matrix is centered using
           the model mean vector (mu)
           
           Returns three numpy objects  
           X:    deflated X matrix
           t:    scores
           d:    distance to model   
        """

        if a >= self.A:     # a is an index staring in 0!
            return (False, 'Too many PC')
            
        nobj,nvar = np.shape (X)

        if a==0:
            X-=self.mu   # centering
            X*=self.wg   # scaling with the same weights
        
        t = np.zeros(nobj,dtype=np.float64)
        d = np.zeros(nobj,dtype=np.float64)
        
        for i in range (nobj):
            obj = X[i,:]

            # obtain scores for object
            t[i] = np.dot(obj,self.p[a])

            # deflates X
            obj -=self.p[a]*t[i]

            # DModX computed after deflating
            d[i] = np.dot(obj.T,obj)
            d[i] = np.sqrt(d[i]/(nvar-a)) # must be divided by SSX/DOF for the model!

        return (True,(X, t, d))

################################################################################

def readData (filename):
    """Reads a numpy X matrix from a file in GOLPE .dat format

       Returns the X matrix as a numpy matrix
    """

    f = open (filename)
    line=f.readline()
    line=f.readline()
    nvar=int(line)
    line=f.readline()
    nobj=int(line)

    X = np.zeros((nobj,nvar),dtype=np.float64)
    for i in range(nobj):
        line = f.readline()
        line = f.readline()
        for j in range(nvar):
            line = f.readline()
            X[i,j]=float(line)

    f.close()
    return X   


if __name__ == "__main__":
    
    # this is only testing code that can be used as an example of use

    # loads data
    X = readData('test01.dat')
    
    mypca = pca ()
    mypca.build(X,targetA=4,autoscale=True)
    mypca.saveModel('modelPCA.npy')

    for a in range (mypca.A):
        print "SSXex %6.4f SSXac %6.4f " % \
              (mypca.SSXex[a]/mypca.SSX, mypca.SSXac[a]/mypca.SSX)

    # reloads the data
    X = readData ('test01.dat')
    nobj,nvar= np.shape(X)

    # creates a new PLS object, reading the model saved above
    pca2 = pca ()
    pca2.loadModel('modelPCA.npy')

    # projects the data on the loaded model
    for i in range(4):
        success, result = pca2.projectPC(X,i)
        if success:
            X, t, dmodx = result
            print t
        else:
            print result
