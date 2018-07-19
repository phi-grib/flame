  
  
  
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

        """Reduced matrices come from deleting cols with low variance??"""
        nvarxb = np.sum(index) # number of informative variables (std> 1e-10)

        #print index
        
        Xb = np.empty((nobj, nvarxb), dtype=np.float64) # New empty matrix with ncols = nvarxb(reduced)
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



###################################

def generateDesignFFD (nvarx, ratio):
    
    ncomb = 1
    R = 0
    targetCom = nvarx * ratio

    for i in range (nvarx):
        ncomb *= 2
        R+=1
        if (ncomb==4096) : break
        if (ncomb>targetCom) : break

##    print 'targetCom: %d' % targetCom, 
##    print 'ncomb: %d' % ncomb, 
##    print 'R: %d' % R 
    
    Dis = np.ones((ncomb,14),dtype=np.int)
    G   = np.zeros((ncomb+1,14),dtype=np.int)    
    lineDis = np.ones((ncomb,nvarx), dtype=np.int)

    # Dis is the full factorial design 2^R located at
    # the beginning of the final design, filling R columns
    
    h = 1
    for y in range(R):
        h*=2
        for w in range (ncomb/h):
            for z in range (h/2):
                Dis[w*h+(h/2)+z][y] = -1

    # G is the "generator": an index of colums of Dis that
    # shoud be multiplied for generating the rest of the design
    # columns
    
    for i in range (1,R+1) :
        G[i][1] = i

    NG = R
    n1 = 1
    n2 = R
    for i in range (1,R+1):
        for j in range (n1,n2+1):
            for h in range (G[j][i]+1,R+1):
                NG+=1
                for f in range (1,G[j][i]+1):
                    G[NG][f] = G[j][f]

                G[NG][i+1]=h

        n1 = n2 + 1
        n2 = NG

    # looks like G is generated in blocks of R elements
    for i in range (1,(NG-R)+1):
        for j in range (1,R+1):
            G[i][j] = G[i+R][j]

    NG -= R

##   Attempt of using C indexing style... failed because the algorithm makes use
##   of the index values to control the loops
    
##    for i in range (R) :
##        G[i,0] = i
##
##    NG = R-1
##    n1 = 0
##    n2 = R-1
##    for i in range (R):
##        for j in range (n1,n2+1):
##            for h in range (G[j,i]+1,R):
##                NG+=1
##                for f in range (G[j,i]):
##                    G[NG,f] = G[j,f]
##                print NG, i
##                G[NG,i+1]=h
##
##        n1 = n2 +1 
##        n2 = NG
##
##
##
##    # looks like G is generated in blocks of R elements
##    for i in range (NG-R):
##        for j in range (R):
##            G[i,j] = G[i+R,j]
##
##    NG -= R
    
    # now we fill the real design
    for i in range (R):
        lineDis[:,i]=Dis[:,i]
    
    for w in range (ncomb):
        disw = Dis[w,:]
        # the next columns are generated using the Gen indexes
        for h in range (nvarx-R):
            nneg = 0
            gg = G[NG-h,:]
            for t in range (1,R+1):
                if disw[gg[t]-1] < 0 : nneg += 1   ## gg indexes must be decreased (-1) because now Dis use C style indexing
                #if disw[gg[t-1]] < 0 : nneg += 1   ## C style indexing
            if nneg%2 : lineDis[w,h+R] = -1
    
    
##    for i in range (ncomb):
##        for j in range (nvarx):
##            print '%2d' % lineDis[i][j],
##        print
    
    return (ncomb, lineDis)

