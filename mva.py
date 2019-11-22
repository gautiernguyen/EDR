# this module codes the Minimum Variance Analysis (MVA) for the magnetic field

import numpy as np

# L,M,N constants
L = 2
M = 1
N = 0



class MVA(object):
    """class for MVA"""

    def __init__(self, B):
        """
        @param B : magnetic field

        """

        B2 = B[:,0]**2 + B[:,1]**2 + B[:,2]**2

        self._B     = B.copy()                             # magnetic field
        #self._B[:,0]/=np.sqrt(B2)
        #self._B[:,1]/=np.sqrt(B2)
        #self._B[:,2]/=np.sqrt(B2)

        #print self._B.shape
        #print np.sqrt(self._B[:,0]**2 + self._B[:,1]**2 + self._B[:,2]**2)
        #print np.sqrt(B[:,0]**2 + B[:,1]**2 + B[:,2]**2)

        self._LMN   = np.ndarray(shape=(3,3))            # LMN matrix
        self._eigen = None


        varM = np.ndarray(shape=(3,3))


        # build the variance matrix <BiBj> - <Bi><Bj>
        for i in np.arange(3):
            for j in np.arange(3):
                varM[i,j] = np.mean(self._B[:,i]*self._B[:,j]) - \
                            np.mean(self._B[:,i])*np.mean(self._B[:,j])


        # find the eigenvalues of varM and the associated eigenvectors
        w,v = np.linalg.eigh(varM)


        # now we want to sort the eigenvalues
        ind         = np.argsort(w) # return the indices that would sort 'w'
        self._eigen = w[ind]

        vt = v.transpose()  # eigh returns v[:,i] associated with w[i]
                            # and we want the line v[i,:] with w[i]
                            # so that vB is Blmn

        # then put the vectors in the same order as the eigenvalues
        for h in np.arange(3) : self._LMN[h,:] = vt[ind[h],:]


        if (self._LMN[N,0] < 0) : # if N points inside the magnetosphere
            self._LMN[N,:] *= -1  # reverse N


        if self._LMN[L,2] < 0:
            self._LMN[L,:] *= -1

        self._LMN[1,0] = self._LMN[0,1]*self._LMN[2,2] - self._LMN[0,2]*self._LMN[2,1]
        self._LMN[1,1] = self._LMN[2,0]*self._LMN[0,2] - self._LMN[2,2]*self._LMN[0,0]
        self._LMN[1,2] = self._LMN[0,0]*self._LMN[2,1] - self._LMN[0,1]*self._LMN[2,0]

            #self._LMN[L,:] *= -1  # reverse L and keep the same M



    #==========================================================
    #==========================================================
    def vec2lmn(self, V):
        """transforms V[:,3] into the LMN coord. syst.
        Creation : 2013-11-12 17:04:58.145602
        """
        vlmn = np.ndarray(V.shape)
        for i in np.arange(3):
            vlmn[:,i] =  self._LMN[i,0]*V[:,0]\
                      +  self._LMN[i,1]*V[:,1]\
                      +  self._LMN[i,2]*V[:,2]


        return vlmn
    #==========================================================



    #==========================================================
    #==========================================================
    def quality(self):
        """returns the quality of MVA as the ratio intermediate/min eigenvalues
        Creation : 2013-11-21 16:18:48.132401
        """
        return self._eigen[1]/self._eigen[0]
    #==========================================================


