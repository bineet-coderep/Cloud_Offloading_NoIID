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

    def getU(self,s):
        '''
        Returns the input vector
        '''

        (K,k,p)=self.getKs(s)

        K_inv=LA.inv(K)

        u_joined=np.matmul(-K_inv,k)
        u_vec=self.decodeU(u_joined)

        #print(u_vec)

        return u_vec

    def getTraj(self,s):
        '''
        Consider the dynamics:
            x'=Ax+Nu+Cs
        '''
        reachList=[self.x0]
        T=len(s)
        U=self.getU(s)
        for i in range(T):
            reach_i=np.matmul(self.A,reachList[-1])+np.matmul(self.B,U[i])+np.matmul(self.C,self.s_oracle[i])
            reachList.append(reach_i)
        return reachList

    def getOracleTraj(self):
        '''
        Returns the Oracle Trajectory
        '''
        reachList=self.getTraj(self.s_oracle)
        return reachList

    def getSiTraj(self,i):
        '''
        Returns the Cloud Trajectory
        '''
        s_i=self.s_i_list[i]
        reachList=self.getTraj(s_i)
        return reachList

    def getPerFromChoice(self,offloadVec):
        '''
        Given an offloadVec, returns the corresponding perception vector
        '''

        perVec=[]
        p=self.C.shape[1]
        for i in range(self.T):
            choice=offloadVec[i]
            # Encode g_per . . .
            pert=self.s_i_list[choice-1][i]
            perVec.append(pert)
        return perVec

    def getOracleStateCost(self,u=-404):

        '''
        Returns cost of the trajectory
        '''
        s=self.s_oracle
        (K,k,indep_u)=self.getKs(self.s_oracle)
        if u==-404:
            u=self.getU(s)
        u_enc=SystemExp.encodeS(u)
        cost_tmp1=np.matmul(np.matmul(u_enc.T,K),u_enc).item()
        cost_tmp2=2*(k.T)
        cost_tmp3=np.matmul(cost_tmp2,u_enc).item()
        stateCost=cost_tmp1+cost_tmp3+indep_u
        return stateCost


    def getStateCost(self,offloadVec,u=-404):

        '''
        Returns cost of the trajectory
        '''
        (K,k,indep_u)=self.getKs(self.s_oracle)
        s=self.getPerFromChoice(offloadVec)
        if u==-404:
            u=self.getU(s)
        u_enc=System.encodeS(u)
        cost_tmp1=np.matmul(np.matmul(u_enc.T,K),u_enc).item()
        cost_tmp2=2*(k.T)
        cost_tmp3=np.matmul(cost_tmp2,u_enc).item()
        stateCost=cost_tmp1+cost_tmp3+indep_u
        return stateCost

    def getR(self):
        '''
        Return:
            blockdiag(R . . . R): mH*mH dimension
        '''
        R_big=LA.block_diag(self.R,self.R)
        for t in range(self.T-2):
            R_big=copy.copy(LA.block_diag(R_big,self.R))
        return R_big

    def getKs(self,s):
        '''
        Return:
            - K
            - k(x_0,s)
            - indep-u
        '''
        (A_list,AB_list,AC_list)=self.getExpList()
        n=self.A.shape[0]
        m=self.B.shape[1]
        H=self.T

        K=np.zeros((m*H,m*H))
        k=np.zeros((m*H,1))
        indep_u=0
        s_enc=System.encodeS(s)

        for i in range(self.T):
            M_i=self.getM(AB_list,i)
            N_i=self.getN(AC_list,i)

            K=copy.copy(K)+(np.matmul(M_i.T,np.matmul(self.Q,M_i)))

            #np.matmul(A_list[i+1],self.x0)
            #print(N_i.shape,s_enc.shape)
            #np.matmul(N_i,s_enc)

            k=copy.copy(k)+np.matmul(M_i.T,np.matmul(self.Q,np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc)))

            indep_u=indep_u+np.matmul((np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc)).T,np.matmul(self.Q,np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc))).item()
        K=self.getR()+copy.copy(K)
        return (K,k,indep_u)

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

    def encodeS(s):
        '''
        Concatinate s vectors
        '''
        return np.vstack(s)

    def decodeU(self,u):
        '''
        given a contatinated u vector, return a list of u
        '''
        u_list=[]
        m=self.B.shape[1]
        u_step=np.zeros((m,1))
        ct=0
        for t in range(self.T):
            for i in range(m):
                u_step[i][0]=u[ct]
                ct=ct+1
            u_list.append(copy.copy(u_step))

        return u_list
