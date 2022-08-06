'''
A class defining a System.
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *

import numpy as np
import random
import scipy.linalg as LA
import copy
import control


from gurobipy import *


class System:
    def __init__(self,A,B,C,Q,R,diffModelsExpectation,diffModelsVariance,x0):
     '''
     Define the dynamics matrices
     '''
     self.A=A
     self.B=B
     self.C=C
     self.diffModelsExpectation=diffModelsExpectation
     self.diffModelsVariance=diffModelsVariance
     self.T=len(diffModelsExpectation[0])
     self.x0=x0
     self.Q=Q
     self.R=R


    def getR(self):
        '''
        Return:
            blockdiag(R . . . R): mH*mH dimension
        '''
        R_big=LA.block_diag(self.R,self.R)
        for t in range(self.T-2):
            R_big=copy.copy(LA.block_diag(R_big,self.R))
        return R_big


    def getKL(self):
        (A_list,AB_list,AC_list)=self.getExpList()
        n=self.A.shape[0]
        m=self.B.shape[1]
        p=self.C.shape[1]
        H=self.T

        K=np.zeros((m*H,m*H))
        L=np.zeros((m*H,p*H))

        for i in range(self.T):
            M_i=self.getM(AB_list,i)
            N_i=self.getN(AC_list,i)
            L=copy.copy(L)+np.matmul(M_i.T,np.matmul(self.Q,N_i))
            K=copy.copy(K)+(np.matmul(M_i.T,np.matmul(self.Q,M_i)))

        K=self.getR()+copy.copy(K)

        return (K,L)


    def getExpList(self):
        '''
        Compute the list [A^i B] and [A^i C]
        '''
        A_list=[np.identity(self.A.shape[0]),self.A]
        AB_list=[self.B]
        AC_list=[self.C]
        for i in range(1,self.T+1):
            AB_list.append(np.matmul(A_list[-1],self.B))
            AC_list.append(np.matmul(A_list[-1],self.C))
            A_list.append(np.matmul(self.A,A_list[-1]))


        return (A_list,AB_list,AC_list)

    def getM(self,AB_list,i):
        '''
        Compute M_i
        '''
        n=self.A.shape[0]
        m=self.B.shape[1]
        H=self.T
        B_list=[]
        for it in range(i+1):
            B_list.append(AB_list[it])
        B_list.reverse()
        M_i_tmp=np.hstack(B_list)
        zeroMi=np.zeros((n,(m*H)-M_i_tmp.shape[1]))
        M_i=np.hstack((M_i_tmp,zeroMi))
        return M_i

    def getN(self,AC_list,i):
        '''
        Compute N_i
        '''
        n=self.A.shape[0]
        m=self.C.shape[1]
        H=self.T
        C_list=[]
        for it in range(i+1):
            C_list.append(AC_list[it])
        C_list.reverse()
        N_i_tmp=np.hstack(C_list)
        zeroNi=np.zeros((n,(m*H)-N_i_tmp.shape[1]))
        N_i=np.hstack((N_i_tmp,zeroNi))
        return N_i
