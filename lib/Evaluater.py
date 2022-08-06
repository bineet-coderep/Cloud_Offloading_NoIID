import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *

import numpy as np
import random
import scipy.linalg as LA
import copy
import control
from lib.MeanVar import *


from gurobipy import *


class Evaluater:
    def __init__(self,sys,alpha=ALPHA,beta=BETA,cost=COST):
        self.sys=sys
        self.alpha=alpha
        self.beta=beta
        self.cost=cost
        p=self.sys.C.shape[1]
        self.s_oracle=[]
        for i in range(self.sys.T):
            s_hat=np.zeros((p,1))
            for i in range(p):
                s_hat[i][0]=np.random.uniform(0,0.002)
            #s_hat[1][0]=0.5
            self.s_oracle.append(s_hat)
        self.s_hat_models=self.instantiateModels()
        self.x0=self.instantiateIC()

    def instantiateIC(self):
        dim=len(self.sys.x0)
        x0=np.zeros((dim,1))
        for i in range(dim):
            x0[i][0]=random.uniform(self.sys.x0[i][0],self.sys.x0[i][1])
        return x0

    def instantiateModels(self):
        '''
        Given an offloadVec, returns the corresponding perception vector
        '''
        s_hat_models=[]
        p=self.sys.C.shape[1]
        W=len(self.sys.diffModelsExpectation)

        for model in range(W):
            s_hat_i=[]
            for timeStep in range(self.sys.T):
                s_hat=np.zeros((p,1))
                exp=self.sys.diffModelsExpectation[model][timeStep]
                var=self.sys.diffModelsVariance[model][timeStep]
                for i in range(p):
                    #print(exp[i])
                    #print(var[i])
                    #val=np.random.normal(exp[i][0],np.sqrt(var[i][0]))
                    val=np.random.normal(exp[i],np.sqrt(var[i]))
                    s_hat[i][0]=val+self.s_oracle[timeStep][i][0]
                s_hat_i.append(s_hat)
            s_hat_models.append(s_hat_i)


        return s_hat_models

    def getPerFromChoice(self,offloadVec):
        '''
        Given an offloadVec, returns the corresponding perception vector
        '''
        perVec=[]
        for i in range(self.sys.T):
            choice=offloadVec[i]
            # Encode g_per . . .
            pert=self.s_hat_models[choice-1][i]
            perVec.append(pert)
        return perVec

    def getOracleControlCost(self):
        '''
        Returns oracle control cost
        '''
        s=self.s_oracle
        (K,k,indep_u)=self.getKs(self.s_oracle)
        u=self.getU(s)
        u_enc=Evaluater.encodeS(u)
        cost_tmp1=np.matmul(np.matmul(u_enc.T,K),u_enc).item()
        cost_tmp2=2*(k.T)
        cost_tmp3=np.matmul(cost_tmp2,u_enc).item()
        stateCost=cost_tmp1+cost_tmp3+indep_u
        return stateCost

    def getControlCost(self,offloadVec):
        '''
        Returns control cost for the offload vector
        '''
        (K,k,indep_u)=self.getKs(self.s_oracle)
        s_hat=self.getPerFromChoice(offloadVec)
        u=self.getU(s_hat)
        #print(u[:3])
        u_enc=Evaluater.encodeS(u)
        cost_tmp1=np.matmul(np.matmul(u_enc.T,K),u_enc).item()
        cost_tmp2=2*(k.T)
        cost_tmp3=np.matmul(cost_tmp2,u_enc).item()
        stateCost=cost_tmp1+cost_tmp3+indep_u
        return stateCost

    def getPerCost(self,offloadVec):
        '''
        Given an `offloadVec` --- A vector of 0-(W-1) depicting offloading choices
        --- returns the model cost
        '''
        modelCost=0
        for c in offloadVec:
            modelCost+=(c*self.cost)
        return modelCost

    def computeReward(self,offloadVec):
        totModelCost=self.getPerCost(offloadVec)
        totStateCost=self.getControlCost(offloadVec)
        totReward=(-self.alpha*totStateCost)+(-self.beta*totModelCost)
        return totReward

    def computeOracleReward(self):
        totModelCost=0
        totStateCost=self.getOracleControlCost()
        totReward=(-self.alpha*totStateCost)+(-self.beta*totModelCost)
        return totReward

    def getKs(self,s):
        '''
        Return:
            - K
            - k(x_0,s)
            - indep-u
        '''
        (A_list,AB_list,AC_list)=self.sys.getExpList()
        n=self.sys.A.shape[0]
        m=self.sys.B.shape[1]
        H=self.sys.T

        K=np.zeros((m*H,m*H))
        k=np.zeros((m*H,1))
        indep_u=0
        s_enc=Evaluater.encodeS(s)

        for i in range(self.sys.T):
            M_i=self.sys.getM(AB_list,i)
            N_i=self.sys.getN(AC_list,i)

            K=copy.copy(K)+(np.matmul(M_i.T,np.matmul(self.sys.Q,M_i)))

            k=copy.copy(k)+np.matmul(M_i.T,np.matmul(self.sys.Q,np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc)))

            indep_u=indep_u+np.matmul((np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc)).T,np.matmul(self.sys.Q,np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc))).item()
        K=self.sys.getR()+copy.copy(K)
        #print(k[0:5])
        #exit()
        return (K,k,indep_u)

    def getU(self,s):
        '''
        Returns the input vector
        '''

        (K,k,p)=self.getKs(s)

        K_inv=LA.inv(K)

        u_joined=np.matmul(-K_inv,k)

        u_vec=self.decodeU(u_joined)

        return u_vec

    def getUMat(self,s):
        '''
        Returns the input vector
        '''

        (K,k,p)=self.getKs(s)

        K_inv=LA.inv(K)

        u_joined=np.matmul(-K_inv,k)

        return u_joined

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
        m=self.sys.B.shape[1]
        u_step=np.zeros((m,1))
        ct=0
        for t in range(self.sys.T):
            for i in range(m):
                u_step[i][0]=u[ct]
                ct=ct+1
            u_list.append(copy.copy(u_step))

        return u_list

    def getTraj(self,offloadVec):
        '''
        Consider the dynamics:
            x'=Ax+Nu+Cs
        '''
        reachList=[self.x0]
        T=len(offloadVec)
        U=self.getUMat(self.getPerFromChoice(offloadVec))
        (A_list,AB_list,AC_list)=self.sys.getExpList()
        for i in range(T):
            partI=np.matmul(A_list[i],self.x0)
            M_i=self.sys.getM(AB_list,i)
            N_i=self.sys.getN(AC_list,i)
            partII=np.matmul(M_i,U)
            #print(N_i.shape,Evaluater.encodeS(self.s_oracle).shape)
            partIII=np.matmul(N_i,Evaluater.encodeS(self.s_oracle))
            reach_i=partI+partII+partIII
            reachList.append(reach_i)
        return reachList

    def getOracleTraj(self):
        '''
        Consider the dynamics:
            x'=Ax+Nu+Cs
        '''
        reachList=[self.x0]
        T=self.sys.T
        U=self.getUMat(self.s_oracle)

        (A_list,AB_list,AC_list)=self.sys.getExpList()
        for i in range(T):
            partI=np.matmul(A_list[i],self.x0)
            M_i=self.sys.getM(AB_list,i)
            N_i=self.sys.getN(AC_list,i)
            partII=np.matmul(M_i,U)
            #print(N_i.shape,Evaluater.encodeS(self.s_oracle).shape)
            partIII=np.matmul(N_i,Evaluater.encodeS(self.s_oracle))
            reach_i=partI+partII+partIII
            reachList.append(reach_i)
        return reachList


class EvaluaterDNN:
    def __init__(self,sys,alpha=ALPHA,beta=BETA,cost=COST):
        self.sys=sys
        self.alpha=alpha
        self.beta=beta
        self.cost=cost
        p=self.sys.C.shape[1]
        self.s_oracle=[]
        self.s_hat_models=[]
        self.x0=self.instantiateIC()

    def instantiateIC(self):
        dim=len(self.sys.x0)
        x0=np.zeros((dim,1))
        for i in range(dim):
            x0[i][0]=random.uniform(self.sys.x0[i][0],self.sys.x0[i][1])
        return x0



    def instantiateModels(self):
        '''
        Given an offloadVec, returns the corresponding perception vector
        '''
        s_hat_models=[]
        p=self.sys.C.shape[1]
        W=len(self.sys.diffModelsExpectation)

        for model in range(W):
            s_hat_i=[]
            for timeStep in range(self.sys.T):
                s_hat=np.zeros((p,1))
                exp=self.sys.diffModelsExpectation[model][timeStep]
                var=self.sys.diffModelsVariance[model][timeStep]
                for i in range(p):
                    #print(exp[i])
                    #print(var[i])
                    #val=np.random.normal(exp[i][0],np.sqrt(var[i][0]))
                    val=np.random.normal(exp[i],np.sqrt(var[i]))
                    s_hat[i][0]=val+self.s_oracle[timeStep][i][0]
                s_hat_i.append(s_hat)
            s_hat_models.append(s_hat_i)


        return s_hat_models

    def getPerFromChoice(self,offloadVec):
        '''
        Given an offloadVec, returns the corresponding perception vector
        '''
        perVec=[]
        for i in range(self.sys.T):
            choice=offloadVec[i]
            # Encode g_per . . .
            pert=self.s_hat_models[choice-1][i]
            perVec.append(pert)
        return perVec

    def getOracleControlCost(self):
        '''
        Returns oracle control cost
        '''
        s=self.s_oracle
        (K,k,indep_u)=self.getKs(self.s_oracle)
        u=self.getU(s)
        u_enc=Evaluater.encodeS(u)
        cost_tmp1=np.matmul(np.matmul(u_enc.T,K),u_enc).item()
        cost_tmp2=2*(k.T)
        cost_tmp3=np.matmul(cost_tmp2,u_enc).item()
        stateCost=cost_tmp1+cost_tmp3+indep_u
        return stateCost

    def getControlCost(self,offloadVec):
        '''
        Returns control cost for the offload vector
        '''
        (K,k,indep_u)=self.getKs(self.s_oracle)
        s_hat=self.getPerFromChoice(offloadVec)
        u=self.getU(s_hat)
        #print(u[:3])
        u_enc=Evaluater.encodeS(u)
        cost_tmp1=np.matmul(np.matmul(u_enc.T,K),u_enc).item()
        cost_tmp2=2*(k.T)
        cost_tmp3=np.matmul(cost_tmp2,u_enc).item()
        stateCost=cost_tmp1+cost_tmp3+indep_u
        return stateCost

    def getPerCost(self,offloadVec):
        '''
        Given an `offloadVec` --- A vector of 0-(W-1) depicting offloading choices
        --- returns the model cost
        '''
        modelCost=0
        for c in offloadVec:
            modelCost+=(c*self.cost)
        return modelCost

    def computeReward(self,offloadVec):
        totModelCost=self.getPerCost(offloadVec)
        totStateCost=self.getControlCost(offloadVec)
        totReward=(-self.alpha*totStateCost)+(-self.beta*totModelCost)
        return totReward

    def computeOracleReward(self):
        totModelCost=0
        totStateCost=self.getOracleControlCost()
        totReward=(-self.alpha*totStateCost)+(-self.beta*totModelCost)
        return totReward

    def getKs(self,s):
        '''
        Return:
            - K
            - k(x_0,s)
            - indep-u
        '''
        (A_list,AB_list,AC_list)=self.sys.getExpList()
        n=self.sys.A.shape[0]
        m=self.sys.B.shape[1]
        H=self.sys.T

        K=np.zeros((m*H,m*H))
        k=np.zeros((m*H,1))
        indep_u=0
        s_enc=EvaluaterDNN.encodeS(s)

        for i in range(self.sys.T):
            M_i=self.sys.getM(AB_list,i)
            N_i=self.sys.getN(AC_list,i)

            K=copy.copy(K)+(np.matmul(M_i.T,np.matmul(self.sys.Q,M_i)))

            #print(N_i.shape,s_enc.shape)

            k=copy.copy(k)+np.matmul(M_i.T,np.matmul(self.sys.Q,np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc)))

            indep_u=indep_u+np.matmul((np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc)).T,np.matmul(self.sys.Q,np.matmul(A_list[i+1],self.x0)+np.matmul(N_i,s_enc))).item()
        K=self.sys.getR()+copy.copy(K)
        #print(k[0:5])
        #exit()
        return (K,k,indep_u)

    def getU(self,s):
        '''
        Returns the input vector
        '''

        (K,k,p)=self.getKs(s)

        K_inv=LA.inv(K)

        u_joined=np.matmul(-K_inv,k)

        u_vec=self.decodeU(u_joined)

        return u_vec

    def getUMat(self,s):
        '''
        Returns the input vector
        '''

        (K,k,p)=self.getKs(s)

        K_inv=LA.inv(K)

        u_joined=np.matmul(-K_inv,k)

        return u_joined

    def encodeS(s):
        '''
        Concatinate s vectors
        '''
        #print(len(s))
        return np.vstack(s)

    def decodeU(self,u):
        '''
        given a contatinated u vector, return a list of u
        '''
        u_list=[]
        m=self.sys.B.shape[1]
        u_step=np.zeros((m,1))
        ct=0
        for t in range(self.sys.T):
            for i in range(m):
                u_step[i][0]=u[ct]
                ct=ct+1
            u_list.append(copy.copy(u_step))

        return u_list

    def getTraj(self,offloadVec):
        '''
        Consider the dynamics:
            x'=Ax+Nu+Cs
        '''
        reachList=[self.x0]
        T=len(offloadVec)
        U=self.getUMat(self.getPerFromChoice(offloadVec))
        (A_list,AB_list,AC_list)=self.sys.getExpList()
        for i in range(T):
            partI=np.matmul(A_list[i],self.x0)
            M_i=self.sys.getM(AB_list,i)
            N_i=self.sys.getN(AC_list,i)
            partII=np.matmul(M_i,U)
            #print(N_i.shape,Evaluater.encodeS(self.s_oracle).shape)
            partIII=np.matmul(N_i,Evaluater.encodeS(self.s_oracle))
            reach_i=partI+partII+partIII
            reachList.append(reach_i)
        return reachList

    def getOracleTraj(self):
        '''
        Consider the dynamics:
            x'=Ax+Nu+Cs
        '''
        reachList=[self.x0]
        T=self.sys.T
        U=self.getUMat(self.s_oracle)

        (A_list,AB_list,AC_list)=self.sys.getExpList()
        for i in range(T):
            partI=np.matmul(A_list[i],self.x0)
            M_i=self.sys.getM(AB_list,i)
            N_i=self.sys.getN(AC_list,i)
            partII=np.matmul(M_i,U)
            #print(N_i.shape,Evaluater.encodeS(self.s_oracle).shape)
            partIII=np.matmul(N_i,Evaluater.encodeS(self.s_oracle))
            reach_i=partI+partII+partIII
            reachList.append(reach_i)
        return reachList
