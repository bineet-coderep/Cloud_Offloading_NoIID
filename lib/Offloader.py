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



class OffloaderExpectation:

    def __init__(self,sys,cost=COST):
        self.sys=sys
        self.cost=cost

    def getRandOffloader(self):
        W=len(self.sys.diffModelsExpectation)
        offloadVec=[random.randint(1,W) for i in range(self.sys.T)]
        return offloadVec


    def getOptOffloader(self,alpha=ALPHA,beta=BETA,cost=COST):
        print("\n\n>> STATUS: Started Computing Optimal Offloading . . .")
        time_taken=time.time()
        offloadOptVec=self.getOptOffloaderChoices(alpha,beta,cost)


        print("\t* Time Taken: ",time.time()-time_taken)
        print(">> STATUS: . . . Completed Computing Optimal Offloading!\n\n")
        return offloadOptVec

    def getOptOffloaderChoices(self,alpha=ALPHA,beta=BETA,cost=COST):
        '''
        Return the optimal choices
        '''
        (K,L)=self.sys.getKL()
        K_inv=LA.inv(K)
        psi=np.matmul(L.T,np.matmul(K_inv,L))


        n=self.sys.A.shape[0]
        m=self.sys.B.shape[1]
        p=self.sys.C.shape[1]
        H=self.sys.T
        W=len(self.sys.diffModelsExpectation) # number of models


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



        s_var_exp=np.zeros((p*H,1),dtype='object') # expectation part

        # Encode s_var_exp
        ct=0
        for i in range(H):
            for j in range(p):
                objMod=0
                for k in range(W):
                    objMod+=(offloadChoiceVars[i][k]*self.sys.diffModelsExpectation[k][i][j])
                s_var_exp[ct]=objMod
                ct=ct+1


        objectiveStateCostExp=np.matmul(s_var_exp.T,np.matmul(psi,s_var_exp)).item() # Encodes the expectation part

        s_var_var=np.zeros((p*H,1),dtype='object') # variance part

        # Encode s_var_var
        ct=0
        for i in range(H):
            for j in range(p):
                objMod=0
                for k in range(W):
                    objMod+=(offloadChoiceVars[i][k]*self.sys.diffModelsVariance[k][i][j])
                s_var_var[ct]=objMod
                ct=ct+1


        boldV=np.diag(psi)
        objectiveStateCostVar=np.matmul(boldV,s_var_var).item()

        objectiveStateCost=objectiveStateCostExp+objectiveStateCostVar

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

    def getOptOffloaderContinous(self,alpha=ALPHA,beta=BETA,cost=COST):
        print("\n\n>> STATUS: Started Computing Optimal Offloading . . .")
        time_taken=time.time()
        offloadOptVec=self.getOptOffloaderChoicesContinous(alpha,beta,cost)


        print("\t* Time Taken: ",time.time()-time_taken)
        print(">> STATUS: . . . Completed Computing Optimal Offloading!\n\n")
        return offloadOptVec

    def getOptOffloaderChoicesContinous(self,alpha=ALPHA,beta=BETA,cost=COST):
        '''
        Return the optimal choices
        '''
        (K,L)=self.sys.getKL()
        K_inv=LA.inv(K)
        psi=np.matmul(L.T,np.matmul(K_inv,L))


        n=self.sys.A.shape[0]
        m=self.sys.B.shape[1]
        p=self.sys.C.shape[1]
        H=self.sys.T
        W=len(self.sys.diffModelsExpectation) # number of models

        expRbt=self.sys.diffModelsExpectation[0]
        varRbt=self.sys.diffModelsVariance[0]
        expCld=self.sys.diffModelsExpectation[W-1]
        varCld=self.sys.diffModelsVariance[W-1]


        model=Model("qp")
        model.params.OutputFlag = 0

        offloadChoiceVars=[]

        for i in range(H):
            name="Ch."+str(i)
            offloadChoiceVars.append(model.addVar(lb=0.0,ub=1.0,name=name,vtype=GRB.CONTINUOUS))



        s_var_exp=np.zeros((p*H,1),dtype='object') # expectation part

        # Encode s_var_exp
        ct=0
        for i in range(H):
            for j in range(p):
                objMod=0
                objMod+=((offloadChoiceVars[i]*expCld[i][j][0])+((1-offloadChoiceVars[i])*expRbt[i][j][0]))
                s_var_exp[ct]=objMod
                ct=ct+1


        objectiveStateCostExp=np.matmul(s_var_exp.T,np.matmul(psi,s_var_exp)).item() # Encodes the expectation part

        s_var_var=np.zeros((p*H,1),dtype='object') # variance part

        # Encode s_var_var
        ct=0
        for i in range(H):
            for j in range(p):
                objMod=0
                objMod+=((offloadChoiceVars[i]*varCld[i][j][0])+((1-offloadChoiceVars[i])*varRbt[i][j][0]))
                s_var_var[ct]=objMod
                ct=ct+1


        boldV=np.diag(psi)
        objectiveStateCostVar=np.matmul(boldV,s_var_var).item()

        objectiveStateCost=objectiveStateCostExp+objectiveStateCostVar

        objectiveModelCost=0
        for i in range(H):
            objectiveModelCost=objectiveModelCost+(offloadChoiceVars[i]*cost)



        objective=(alpha*objectiveStateCost)+(beta*objectiveModelCost)

        # Set Objective
        model.setObjective(objective,GRB.MINIMIZE)
        model.optimize()


        choicesOpt=[]
        for v in model.getVars():
            if v.varName[:2]=="Ch":
                #print('%s %g' % (v.varName, v.x))
                choicesOpt.append(v.x)



        return choicesOpt
