import numpy as np

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

if __name__ == "__main__":

    generateDesignFFD (500,2)    
