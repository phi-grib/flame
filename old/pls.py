# -*- coding: utf-8 -*-

##    Description    PLS toolkit using NIPALS algorithm 
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
import cProfile

from scipy import stats
from scipy.stats import t

import sys
from scale import center, scale
from qualit import *
from utils import updateProgress
from design import generateDesignFFD
class pls:

    def __init__ (self):
        self.X = None
        self.Y = None
        
        self.Am = 0  # model dimensionality
        self.Av = 0  # number of LV validated. Notice that Av <= Am
        self.nobj = 0
        self.nvarx = 0
        
        self.mux = None
        self.muy = None

        self.wgx = None

        self.autoscale = False
        
        self.t = []  # scores
        self.p = []  # loadings
        self.w = []  # weights
        self.c = []  # inner relationship
        self.SSXex = []   # SSX explained
        self.SSXac = []   # SSX accumulated
        self.SSYex = []   # SSY explained
        self.SSYac = []   # SSY accumulated
        self.SDEC = []    # SD error of the calculations
        self.dmodx = []   # distance to model

        self.cutoff = []
        
        self.TP = []
        self.TN = []
        self.FP = []
        self.FN = []

        self.TPpred = []
        self.TNpred = []
        self.FPpred = []
        self.FNpred = []
        
        self.SSY  = []    # SSY explained
        self.SDEP = []    # SD error of the predictions
        self.Q2   = []    # cross-validated R2

    def saveModel(self,filename):
        """Saves the whole model to a binary file in numpy .npy format

        """

        f = file(filename,'wb')

        np.save(f,self.Am)
        np.save(f,self.Av)
        np.save(f,self.nobj)
        np.save(f,self.nvarx)
        
        np.save(f,self.mux)
        np.save(f,self.muy)
        np.save(f,self.wgx)

        np.save(f,self.autoscale)
        
        for a in range(self.Am):
            np.save(f,self.t[a])
            np.save(f,self.p[a])
            np.save(f,self.w[a])
            np.save(f,self.c[a])
            np.save(f,self.SSXex[a])
            np.save(f,self.SSXac[a])
            np.save(f,self.SSYex[a])
            np.save(f,self.SSYac[a])
            np.save(f,self.SDEC[a])
            np.save(f,self.dmodx[a])
            
            np.save(f,self.cutoff[a])

            np.save(f,self.TP[a])
            np.save(f,self.TN[a])
            np.save(f,self.FP[a])
            np.save(f,self.FN[a])

            np.save(f,self.TPpred[a])
            np.save(f,self.TNpred[a])
            np.save(f,self.FPpred[a])
            np.save(f,self.FNpred[a])

        for a in range(self.Av):
            np.save(f,self.SSY[a])
            np.save(f,self.SDEP[a])
            np.save(f,self.Q2[a])
        
        f.close()

            
    def loadModel(self,filename):
        """Loads the whole model from a binary file in numpy .npy format

        """

        f = file(filename,'rb')
        
        self.Am = np.load(f)
        self.Av = np.load(f)
        self.nobj = np.load(f)
        self.nvarx = np.load(f)
        
        self.mux = np.load(f)
        self.muy = np.load(f)
        self.wgx = np.load(f)

        self.autoscale = np.load(f)
        
        for a in range(self.Am):
            self.t.append (np.load(f))
            self.p.append (np.load(f))
            self.w.append (np.load(f))
            self.c.append (np.load(f))
            self.SSXex.append (np.load(f))
            self.SSXac.append (np.load(f))
            self.SSYex.append (np.load(f))
            self.SSYac.append (np.load(f))
            self.SDEC.append (np.load(f))
            self.dmodx.append (np.load(f))
            
            self.cutoff.append (np.load(f))

            self.TP.append (np.load(f))
            self.TN.append (np.load(f))
            self.FP.append (np.load(f))
            self.FN.append (np.load(f))
            
            self.TPpred.append (np.load(f))
            self.TNpred.append (np.load(f))
            self.FPpred.append (np.load(f))
            self.FNpred.append (np.load(f))
            
        for a in range(self.Av): 
            self.SSY.append (np.load(f))
            self.SDEP.append (np.load(f))
            self.Q2.append (np.load(f))

        f.close()

    def loadDistiled(self,filename):

        f = file(filename,'r')

        self.Am = int (f.readline())
        self.nvarx = int (f.readline())
        self.autoscale = bool (f.readline())

        t = np.loadtxt (f)
        c = 0

        self.muy = t[c]
        c += 1

        self.mux = np.zeros(self.nvarx,dtype=np.float64)
        for i in range (self.nvarx):
            self.mux[i] = t[c]
            c += 1

        self.wgx = np.zeros(self.nvarx,dtype=np.float64)
        for i in range (self.nvarx):
            self.wgx[i] = t[c]
            c += 1

        for a in range (self.Am):
            p = np.zeros(self.nvarx,dtype=np.float64)
            for i in range (self.nvarx):
                p[i] = t[c]
                c += 1
            self.p.append(p)
                
            w = np.zeros(self.nvarx,dtype=np.float64)
            for i in range (self.nvarx):
                w[i] = t[c]
                c += 1
            self.w.append(w)

        for a in range(self.Am):
            self.c.append(t[c])
            c += 1

        for a in range(self.Am): 
            self.cutoff.append(t[c])
            c += 1            

        f.close()

##        f = file(filename,'rb')
##        
##        self.Am = np.load(f)
##        self.nvarx = np.load(f)
##        self.autoscale = np.load(f)
##
##        self.muy = np.load(f)
##        
##        self.mux = np.load(f)
##        self.wgx = np.load(f)
##       
##        for a in range(self.Am):
##            self.p.append (np.load(f))
##            self.w.append (np.load(f))
##            
##            self.c.append (np.load(f))
##            self.cutoff.append (np.load(f))
##
##        f.close()

    def saveDistiled (self, filename):

##        f = file(filename,'wb')
##
##        np.save(f,self.Am)
##        np.save(f,self.nvarx)
##        np.save(f,self.autoscale)
##        
##        np.save(f,self.muy)
##        
##        np.save(f,self.mux)
##        np.save(f,self.wgx)
##        
##        for a in range(self.Am):
##            np.save(f,self.p[a])
##            np.save(f,self.w[a])
##            
##            np.save(f,self.c[a])
##            np.save(f,self.cutoff[a])
##        
##        f.close()

        f = file (filename,'w')

        t1 = np.zeros(3, dtype=np.int16)
        t1[0] = self.Am
        t1[1] = self.nvarx
        t1[2] = self.autoscale       
        np.savetxt(f,t1,fmt='%d')

        t2 = np.zeros(1, dtype=np.float64)
        t2 [0] = self.muy
        np.savetxt(f,t2,fmt='%f')
        
        np.savetxt(f,self.mux,fmt='%f')
        np.savetxt(f,self.wgx,fmt='%f')
        
        for a in range(self.Am):
            np.savetxt(f,self.p[a],fmt='%f')
            np.savetxt(f,self.w[a],fmt='%f')
            
        np.savetxt(f,self.c,fmt='%f')
        np.savetxt(f,self.cutoff,fmt='%f')

        f.close ()
        
    def build (self, X, Y, targetA=0, targetSSX=0.0, autoscale=False):
        """Build a new PLS model with the X and Y numpy matrice provided using NIPALS algorithm

           The dimensionality of the model can be defined either providing
           1. directly the number of LV to extract (targetA)
           2. the fraction of SSX that the model will explain (targetSSX)

           The X and Y matrices are centered but no other scaling transform is applied

           Does not return anything, but updates internals vectors and variables
        """
        nobj, nvarx= np.shape(X)

##        for i in range (nobj):
##            for j in range (nvarx):
##                print X[i,j],
##            print

        self.nobj = nobj
        self.nvarx = nvarx
        self.X = X.copy()
        self.Y = Y.copy()

        self.X, self.mux = center(self.X)
        self.Y, self.muy = center(self.Y)
        self.X, self.wgx = scale(self.X, autoscale)

##        self.mux = mux
##        self.muy = muy
##        self.wgx = wgx

        self.autoscale = autoscale
        
        SSXac=0.0
        SSYac=0.0

        SSX0,SSY0, null = self.computeSS(self.X,self.Y)
        
        SSXold=SSX0
        SSYold=SSY0

        

        a=0
        while True:
            t, p, w, c = self.extractLV(self.X, self.Y)
                
            self.t.append(t) 
            self.p.append(p)
            self.w.append(w)
            self.c.append(c)
            
            self.X, self.Y = self.deflateLV(self.X, self.Y, t, p, c)
            
            SSXnew, SSYnew, dmodx = self.computeSS(self.X, self.Y)

            SSXex = (SSXold-SSXnew)/SSX0
            SSXac+=SSXex

            SSYex = (SSYold-SSYnew)/SSY0
            SSYac+=SSYex

            SDEC = np.sqrt(SSYnew/nobj)

            dof = nvarx-a
            if dof <= 0 : dof = 1
            dmodx = [np.sqrt(d/dof) for d in dmodx] 

            SSXold=SSXnew
            SSYold=SSYnew

            self.SSXex.append(SSXex)
            self.SSXac.append(SSXac)
            self.SSYex.append(SSYex)
            self.SSYac.append(SSYac)
            self.SDEC.append(SDEC)
            self.dmodx.append(dmodx)
            
            a+=1
                
            if targetA>0:
                if a==targetA : break

            if targetSSX>0.0:
                if SSXac>targetSSX: break
                # prevents to extract a meaningless number of LV
                if a > min (20,nobj/5) : break 

        self.Am=a
            
        # NIPALS is destructive, so we must retrieve X and Y from original data for validation
        self.X = X.copy()
        self.Y = Y.copy()
        
        self.cutoff = np.zeros(self.Am, dtype=np.float64)
        self.TP = np.zeros(self.Am)
        self.TN = np.zeros(self.Am)
        self.FP = np.zeros(self.Am)
        self.FN = np.zeros(self.Am)

        self.TPpred = np.zeros(self.Am)
        self.TNpred = np.zeros(self.Am)
        self.FPpred = np.zeros(self.Am)
        self.FNpred = np.zeros(self.Am)

    def validateLOO (self, A, gui=False):
        """ Validates A dimensions of an already built PLS model, using Leave-One-Out cross-validation

            Returns nothing. The results of the cv (SSY, SDEP and Q2) are stored internally
        """

        if self.X == None or self.Y == None:
            return 
        
        X = self.X
        Y = self.Y     

        nobj,nvarx = np.shape (X)

        SSY0 = 0.0
        for i in range (nobj):
            SSY0+=np.square(Y[i]-np.mean(Y))

        SSY = np.zeros(A,dtype=np.float64)
        YP = np.zeros ((nobj,A+1),dtype=np.float64)

        if gui: updateProgress (0.0)
        
        for i in range (nobj):
            
            # build reduced X and Y matrices removing i object
            Xr = np.delete(X,i,axis=0)
            Yr = np.delete(Y,i)

            Xr,muxr = center(Xr)
            Xr,wgxr = scale (Xr, self.autoscale)
           
            Yr,muyr = center(Yr)

            xp = np.copy(X[i,:])
            
            xp -= muxr
            xp *= wgxr
            
            # predicts y for the i object, using A LV
            yp = self.getLOO(Xr,Yr,xp,A)      
            yp += muyr

            # updates SSY with the object i errors
            YP[i,0]=Y[i]
            
            for a in range(A):
                SSY[a]+= np.square(yp[a]-Y[i])
                YP[i,a+1]=yp[a]

            if gui : updateProgress (float(i)/float(nobj))

        if gui : print
        
        self.SSY  = SSY        
        self.SDEP = [np.sqrt(i/nobj) for i in SSY]
        self.Q2   = [1.00-(i/SSY0) for i in SSY]
        
        self.Av = A

        return (YP)


    def project (self, x, A):
        """projects query object x into current model using A LV

           Returns
           y:    vector of predicted Y values using growing number of LV
           t:    vector of scores
           d:    SSX for every dimension
        """

        if A > self.Am:
            return (False, 'Too many LV')
                
        x-=self.mux
        x*=self.wgx

        y=np.zeros(A,dtype=np.float64)
        t=np.zeros(A,dtype=np.float64)
        d=np.zeros(A,dtype=np.float64)

        yp = 0.0
        for a in range (A):        
            t[a] = np.dot(x,self.w[a])
            yp += t[a]*self.c[a]
            y[a]= yp
            x -= self.p[a]*t[a]
            dof = (self.nvarx-a)
            if dof <= 0 : dof = 1
            d[a] = np.sqrt(np.dot(x,x)/dof) 

        y+=self.muy

        return (True, (y, t, d))

        
    def extractLV (self, X, Y):
        """Extracts a single LV from the provided X and Y matrices using NIPALS algorithm

           This method assumes that both X and Y are centered. No deflation is applied
           
           Returns
           t:    vector of scores
           p:    vector of loadings
           w:    vector of weights
           c:    inner relationship
        """
        
        nobj,nvarx = np.shape (X)
        w = np.zeros(nvarx, dtype=np.float64)
        p = np.zeros(nvarx, dtype=np.float64)
        t = np.zeros(nobj , dtype=np.float64)

        uu = np.dot(Y.T,Y)
        for j in range(nvarx):
            w[j] = np.dot(Y,X[:,j])/uu
            
        ww = np.sqrt(np.dot(w,w))
        if ww>1e-9 : w/=ww

        for i in range(nobj):
            t[i] = np.dot(w,X[i,:])

        tt = np.dot(t,t)

        if (tt>1e-9):
            for j in range(nvarx):
                p[j] = np.dot(t,X[:,j])/tt

            c = np.dot(t,Y)/tt
        else:
            c = 0.00

        return t, p, w, c


    def computeSS (self, X, Y):
        """Computes the Sum-Of-Squares for provided X and Y matrices

           Returns
           SSX:    sum of squates of the X matrix
           SSY:    sum of squares of the Y matrix
           d:      vector with the SSX for every object 
        """
        
        nobj,nvarx = np.shape (X)
        
        SSX=SSY=0.0

        d = np.zeros(nobj,dtype=np.float64)
        
        for i in range (nobj):
            d[i] = np.dot(X[i,:],X[i,:])
            SSX += d[i]
            SSY += np.square(Y[i])
            
        return SSX, SSY, d


    def deflateLV (self, X, Y, t, p, c):
        """Deflates both the X and Y matrices, using the provided t, p and c vectors

           Returns deflated X and Y
        """
        
        nobj,nvarx = np.shape (X)
        
        for i in range (nobj):
            X[i,:] -= (t[i]*p)
            Y[i] -= t[i]*c
        
        return X,Y


    def getLOO (self, X, Y, x, A):
        """Builds a model of A dimension with the provided X and Y matrices, yielding a prediction y for the query object x.
           Typically used as inner loop in LOO CV method.

           Notice that both X and Y must be centered, while x must have been centered with the model averages

           Returns the predicted y value for the query object x
        """
        
        y = np.zeros(A,dtype=np.float64)
        
        for a in range(A):
            t, p, w, c = self.extractLV(X, Y)
            if a>0 : y[a]=y[a-1]
           
            tt = np.dot(x,w)    
            y[a] += tt*c
            x -= p*tt
            
            X, Y = self.deflateLV(X, Y, t, p, c)
        return y

    def recalculate (self):
        yr = np.zeros ((self.nobj,self.Am+1),dtype=np.float64)
        for i in range (self.nobj):
            # self.project could be destructive, since the X vector is deflated, so make sure to copy!
            success, result = self.project(self.X[i,:].copy(),self.Am) # just for final #LV
            yr[i,0]=self.Y[i]
            if success:
                yr[i,1:] = result[0]        
        return yr

    def calcConfussion (self, cutoff, ycutoff = 0.5):

        by = []
        yr = self.recalculate()
        for i in range (self.nobj):
            by.append (yr[i][0] > ycutoff) # yr[0] is the experimental Y
            
        for a in range (self.Am):

            TP=TN=FP=FN=0

            for i in range(self.nobj):                
                if by[i]:
                    if yr[i][a+1] > cutoff:
                        TP+=1
                    else:
                        FN+=1
                else:
                    if yr[i][a+1] > cutoff:
                        FP+=1
                    else:
                        TN+=1

##            sens = sensitivity (TP, FN)
##            spec = specificity (TN, FP)
##
##            print 'rec  sens %f' % sens
##            print 'rec  spec %f' % spec    

            self.cutoff[a] = cutoff
            self.TP[a] = TP
            self.TN[a] = TN
            self.FP[a] = FP
            self.FN[a] = FN

    def predConfussion (self, ycutoff = 0.5):

        by = []
        yp = self.validateLOO(self.Am, gui=True)

        for i in range (self.nobj):
            by.append (yp[i][0] > ycutoff) # yp[0] is the experimental Y
            
        for a in range (self.Am):

            TP=TN=FP=FN=0

            for i in range(self.nobj):                
                if by[i]:
                    if yp[i][a+1] > self.cutoff[a]:
                        TP+=1
                    else:
                        FN+=1
                else:
                    if yp[i][a+1] > self.cutoff[a]:
                        FP+=1
                    else:
                        TN+=1

##            sens = sensitivity (TP, FN)
##            spec = specificity (TN, FP)
##
##            print 'pred sens %f' % sens
##            print 'pred spec %f' % spec           

            self.TPpred[a] = TP
            self.TNpred[a] = TN
            self.FPpred[a] = FP
            self.FNpred[a] = FN

        # return a binary (0 = False, 1 = True) array for being processed in ADAN
        ypbin = np.zeros (self.nobj,dtype=np.float64)
        
        for i in range(self.nobj):
            if (yp[i][-1] > self.cutoff[-1]) : ypbin[i] = 1.0 
                
        return (ypbin)
    
                 
    def calcOptCutoff (self, ycutoff = 0.5, nsteps = 100):

        by = []
        yr = self.recalculate()
        for i in range (self.nobj):
            by.append (yr[i][0] > ycutoff) # yr[0] is the experimental Y
            
        for a in range (self.Am):
            bestv  = 1.0e20
            bestc  = 0.0           
            cutoff = 0.0
            bTP=bTN=bFP=bFN=0

            for step in range (nsteps):
                cutoff+=1.0/nsteps
                TP=TN=FP=FN=0

                for i in range(self.nobj):                
                    if by[i]:
                        if yr[i][a+1] > cutoff:
                            TP+=1
                        else:
                            FN+=1
                    else:
                        if yr[i][a+1] > cutoff:
                            FP+=1
                        else:
                            TN+=1

                sens = sensitivity (TP, FN)
                spec = specificity (TN, FP)
                
                if abs(sens-spec) < bestv :
                    bestv = abs(sens-spec)
                    bestc = cutoff
                    bTP = TP
                    bTN = TN
                    bFP = FP
                    bFN = FN

            self.cutoff[a] = bestc
            self.TP[a] = bTP
            self.TN[a] = bTN
            self.FP[a] = bFP
            self.FN[a] = bFN
        
        #return (bestc, (bTP,bTN,bFP,bFN))

    def varSelectionFFD (self, X, Y , A, autoscale=False, gui=True):

        # TODO : set dummyStep and ratio as tunable parameters
        
        dummyStep = 4.0
        ratio     = 2.0

        # TODO : check the number of X variables. FFD is not suitable for very large X matrices

        # build a X reduced matrix Xr
        nobj, nvarx = np.shape (X)
        nvarxOri = nvarx
        index = np.ones(nvarx,dtype=np.int)
        st = np.std (X, axis=0, ddof=1)
        for i in range (nvarx):
            if  st[i] < 1e-10:
                index[i] = 0  # set to 0 to allow creation of reduced matrices
        nvarxb = np.sum(index)

        #print index
        
        Xb = np.empty((nobj, nvarxb), dtype=np.float64)
        k=0
        for i in range (nvarx):
            if index[i]>0:
                Xb[:,k]=X[:,i]
                k+=1
        
        nobj, nvarx = np.shape (Xb)
        ndummy = int (np.floor(nvarx/dummyStep))              # number of dummy variables
        nvarxm = nvarx + ndummy                               # length of expanded vector
        ncomb, design  = generateDesignFFD (nvarxm, ratio)    # ncomb is the number of reduced models to be generated
                                                              # design is the matrix that designates is every x variable
                                                              # is in/out of the design matrix
        # print nvarx, ndummy, nvarxm, ncomb

        # obtain first estimation of Y std error
        SSY0 = 0.0
        for i in range (nobj):
            SSY0+=np.square(Y[i]-np.mean(Y))
        SDEP0 = np.sqrt(SSY0/float(nobj))
        SDEP0x10 = 10.0 * SDEP0

        # initializes effects
        effect  = np.zeros(nvarxm,dtype=np.float64)
        xdesign = np.zeros(nvarx ,dtype=np.int)

        # set common model stuff
        self.autoscale = autoscale
        self.Y = Y.copy()

        if gui: updateProgress (0.0)
        
        for i in range(ncomb):

            # extract x design line (not considering dummies)            
            k=0
            for j in range (nvarxm):
                if j%(dummyStep+1) :         # non-dummy var
                    xdesign[k]=design[i][j]
                    k+=1
                    
            nvarxr = int(np.sum(xdesign>0))

            # if this design line contains few x vars skip the model validation
            if nvarxr <= (A+1) : continue
            
            # build a X reduced matrix Xr
            Xr = np.empty((nobj, nvarxr), dtype=np.float64)
            k=0
            for j in range (nvarx):
                if xdesign[j]>0:
                    Xr[:,k]=Xb[:,j]
                    k+=1

            # set the reduced matrix as model matrix and validate
            self.X = Xr.copy()
            self.validateLOO (A)
            
            # accumulate the min SDEP to a effect vector for every variable (including dummies)
            minSDEP = 2.0e10
            for a in self.SDEP:
                if a < minSDEP : minSDEP = a

            if minSDEP > SDEP0x10:
                minSDEP = SDEP0
            
            effect += design[i]*minSDEP

            if gui: updateProgress (float(i)/float(ncomb))

        # calculate effects
        effect /= (ncomb/2)
        
        # compute dummy effects
        dummyEffect = 0.00
        dummyMean = 0.00
        k  = 0

        for i in range(nvarxm):
            if not (i%(dummyStep+1)) :   # dummy var
                dummyMean+=effect[i]
        dummyMean/=ndummy

        for i in range(nvarxm):
            if not (i%(dummyStep+1)) :   # dummy var
                dummyEffect+=np.square(effect[i]-dummyMean)
                ##dummyEffect+=np.square(effect[i])                 ## old version: assuming mean of zero (?) 
                ##td+=1
            else :
                effect[k]=effect[i]
                k+=1

        if dummyEffect > 1e-6:
            dummySD = np.sqrt(dummyEffect/ndummy)
        else :
            dummySD = 0.001
            
        # compare with critical T values (two tail, 95%)    
        t = stats.t.ppf(0.9725,ndummy-1)
        effectCutoff = t * dummySD
        
        res = np.ones(nvarx,dtype=np.int)          # fixed (default)      
        for i in range(nvarx):
            if np.abs(effect[i]) < effectCutoff:   # uncertain
                res[i] = 2
            elif effect[i] > 0 :
                res[i] = 0                         # excluded

        #print res

        # map the result in a vector representing the full, original X
        resExp = np.ones(nvarxOri,dtype=np.int)
        k = 0
        for i in range (nvarxOri):
            if index[i]==0:
                resExp[i] = 0       # these were already excluded or are inactive variables
            else :
                resExp[i] = res[k]
                k += 1
        
        return resExp, np.sum(res==0)

    def excludeVar (self, X, res):
        X[:,res==0]=0.000   
        return X
          

################################################################################################


def readData (filename):
    """Reads numpy X and Y matrices from a file in GOLPE .dat format, asuming a single Y value at the end

       Returns X and Y as a numpy matrices
    """

    f = open (filename)
    line=f.readline()
    line=f.readline()
    nvar=int(line)
    line=f.readline()
    nobj=int(line)

    X = np.zeros((nobj,nvar-1),dtype=np.float64)
    Y = np.zeros(nobj,dtype=np.float64)
    for i in range(nobj):
        line = f.readline()
        line = f.readline()
        for j in range(nvar-1):
            line = f.readline()
            X[i,j]=float(line)
        line = f.readline()
        Y[i]=float(line)

    f.close()
    return X, Y



if __name__ == "__main__":
    # this is only testing code that can be used as an example of use

    # loads data
    #X, Y = readData ('Biopsycho_2A_activity.dat')
    X, Y = readData ('data02.dat')
    #X, Y = readData ('xanthines.dat')

    #testType = 'PLS' 
    testType = 'FFD'

    if testType == 'PLS' :
        # builds a PLS model
        mypls = pls ()
        mypls.build(X,Y,targetA=5,autoscale=False)     
        mypls.validateLOO(5, gui=False)
        mypls.saveModel('modelPLS.npy')

        # everything complete, print the results
        for a in range (mypls.Am):
            print "SSXex %6.4f SSXac %6.4f " % \
                  (mypls.SSXex[a], mypls.SSXac[a]),
            print "SSYex %6.4f SSYac %6.4f SDEC %6.4f" % \
                  (mypls.SSYex[a], mypls.SSYac[a], mypls.SDEC[a])
         
        for a in range (mypls.Av):
            print 'A:%2d  SSY: %6.4f Q2: %6.4f SDEP: %6.4f' % \
                  (a+1,mypls.SSY[a],mypls.Q2[a],mypls.SDEP[a])
            
    elif testType == 'FFD' :

        FFDcycle = 1
        excludeVars = True

        FFDi = 0
        Auto = False
        mypls = pls ()
        
        while (True):

            # exclude variables
            if (excludeVars):
                #cProfile.run ('res, nexcluded = mypls.varSelectionFFD(X,Y,2,autoscale=Auto)')
                
                res, nexcluded = mypls.varSelectionFFD(X,Y,2,autoscale=Auto)
                X              = mypls.excludeVar(X,res)

                print '\n', nexcluded, ' var excluded'

                mypls.build(X,Y,targetA=5,autoscale=Auto)
                mypls.validateLOO(5, gui=False)
                for a in range (mypls.Av):
                    print 'A:%2d  SSY: %6.4f Q2: %6.4f SDEP: %6.4f' % \
                          (a+1,mypls.SSY[a],mypls.Q2[a],mypls.SDEP[a])
                
                if (nexcluded==0)     : break
                                
                FFDi += 1
                if (FFDi >= FFDcycle) : break

            else :
                break;


##        # builds a PLS model
##        mypls.build(X,Y,targetA=5,autoscale=Auto)
##
##        # validate the model
##        mypls.validateLOO(5, gui=True)
##
##        for a in range (mypls.Av):
##            print 'A:%2d  SSY: %6.4f Q2: %6.4f SDEP: %6.4f' % \
##                  (a+1,mypls.SSY[a],mypls.Q2[a],mypls.SDEP[a])
        

##    # reloads the data
##    x, y = readData ('xanthines.dat')
##    nobj,nvarx= np.shape(x)
##
##    # creates a new PLS object, reading the model saved above
##    pls2 = pls ()
##    pls2.loadModel('modelPLS.npy')
##
##    # projects the data on the loaded model
##    for i in range(nobj):
##        success, result = pls2.project(x[i,:],3)
##        if success:
##            yp, tp, dmodx = result
##            print yp, tp, dmodx[0], pls2.dmodx[0][i]
##            #print pls2.dmodx[0][i]
##        else:
##            print result
  
