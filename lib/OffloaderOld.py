'''
Provides APIs to find out the optimal trajectory,
and several other trajectories
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *
from lib.System import *
import random
import scipy.linalg as LA
import numpy as np
from gurobipy import *
import time



class OffloaderDiscExp:

    def __init__(self,sys,cost=COST):
        self.sys=sys
        self.cost=cost

    def getRandOffloader(self):
        W=len(self.sys.diffModelsExpectation)
        offloadVec=[random.randint(1,W) for i in range(self.sys.T)]
        s_rand=self.sys.getPerFromChoice(offloadVec)

        dictRandTraj={
        0: offloadVec,
        1: s_rand,
        }
        return dictRandTraj


    def getModelCost(self,offloadVec):
        '''
        Given an `offloadVec` --- A vector of 0-(W-1) depicting offloading choices
        --- returns the model cost
        '''
        modelCost=0
        for c in offloadVec:
            modelCost+=(c*self.cost)
        return modelCost

    def computeReward(self,offloadVec,alpha=ALPHA,beta=BETA):
        totModelCost=self.getModelCost(offloadVec)
        totStateCost=self.sys.getStateCost(offloadVec)
        #print(totModelCost,totStateCost)
        #print(ALPHA,BETA)
        totReward=(-alpha*totStateCost)+(-beta*totModelCost)
        return totReward

    def computeOracleReward(self,alpha=ALPHA,beta=BETA):
        totModelCost=0
        totStateCost=self.sys.getOracleStateCost()
        #print(totModelCost,totStateCost)
        #print(totModelCost,totStateCost)
        #print(ALPHA,BETA)
        totReward=(-alpha*totStateCost)+(-beta*totModelCost)
        return totReward

    def getOptOffloader(self,alpha=ALPHA,beta=BETA,cost=COST):
        print("\n\n>> STATUS: Started Computing Optimal Offloading . . .")
        time_taken=time.time()
        offloadOptVec=self.getOptOffloaderChoices(alpha,beta,cost)
        s_opt=self.sys.getPerFromChoice(offloadOptVec)

        dictOptTraj={
        0: offloadOptVec,
        1: s_opt,
        }
        print("\t* Time Taken: ",time.time()-time_taken)
        print(">> STATUS: . . . Completed Computing Optimal Offloading!\n\n")
        return dictOptTraj

    def getOptOffloaderChoices(self,alpha=ALPHA,beta=BETA,cost=COST):
        '''
        Return the optimal choices
        '''
        (K,L)=self.sys.getKL()
        K_inv=LA.inv(K)
        psi=np.matmul(L.T,np.matmul(K_inv,L))
        #print(psi)
        #print()
        #print(psi.shape)

        n=self.sys.A.shape[0]
        m=self.sys.B.shape[1]
        p=self.sys.C.shape[1]
        H=self.sys.T
        W=len(self.sys.s_i_list) # number of models

        s=self.sys.s_oracle

        model=Model("miqp")
        model.params.OutputFlag = 0

        offloadChoiceVars=np.zeros((H,W),dtype='object')


        # Encode choice variables
        for i in range(H):
            name="Ch."+str(i)
            for j in range(W):
                offloadChoiceVars[i][j]=model.addVar(name=name+","+str(j),vtype=GRB.BINARY)

        # Constraintes on model selection: select only one model at a time
        for i in range(H):
            objSum=0
            for j in range(W):
                objSum+=offloadChoiceVars[i][j]
            model.addConstr(objSum==1,name="ChoiceConstr."+str(i))



        s_var=np.zeros((p*H,1),dtype='object')

        # Encode s_var
        ct=0
        for i in range(H):
            for j in range(p):
                objMod=0
                for k in range(W):
                    objMod+=(offloadChoiceVars[i][k]*self.sys.s_i_list[k][i][j])
                obj=objMod-self.sys.s_oracle[i][j][0]
                s_var[ct]=obj
                ct=ct+1


        objectiveStateCost=np.matmul(s_var.T,np.matmul(psi,s_var)).item()

        objectiveModelCost=0
        for i in range(H):
            for j in range(W):
                objectiveModelCost=objectiveModelCost+(offloadChoiceVars[i][j]*(j+1)*cost)



        objective=(alpha*objectiveStateCost)+(beta*objectiveModelCost)

        # Set Objective
        model.setObjective(objective,GRB.MINIMIZE)
        model.optimize()


        '''choicesOpt=[]
        for v in model.getVars():
            if v.varName[:2]=="Ch":
                print('%s %g' % (v.varName, v.x))
                choicesOpt.append(v.x)'''

        choicesOpt=[]
        for i in range(H):
            for j in range(W):
                if offloadChoiceVars[i][j].x==1:
                    choicesOpt.append(j+1)


        return choicesOpt
