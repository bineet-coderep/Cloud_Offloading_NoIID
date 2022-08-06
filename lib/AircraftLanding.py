'''
Provides an example Aircraft Landing System.
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *
from lib.System import *
from lib.MeanVar import *

import numpy as np
import random


class AircraftLanding:

    def getSimpleSystem(x0,T):
        '''
        Define the dynamics matrices
        '''
        K=-0.95
        Ts=2.5
        xi=0.5
        omega=1
        V=20
        a22=(1/Ts)
        a23=(V/Ts)
        a42=(1/(V*Ts))-((2*xi*omega)/(V*Ts))+((omega*omega)/V)
        a43=((2*xi*omega)/(Ts))-(omega*omega)-(1/(Ts*Ts))
        a44=(1/Ts)-(2*xi*omega)
        b4=(omega*omega)/K
        A2=np.array([
        [0,1,0,0],
        [0,a22,a23,0],
        [0,0,0,1],
        [0,a42,a43,a44]
        ])
        B2=np.array([
        [0],
        [0],
        [0],
        [b4]
        ])
        h=0.01
        A=np.add(np.identity(A2.shape[0]),h*A2)
        B=h*B2
        C2=np.array([
        [1,0],
        [0,0],
        [0,1],
        [0,0]
        ])
        C=np.array([
        [0],
        [0],
        [1],
        [0]
        ])

        Q=1*np.identity(A.shape[0])
        #Q[0][0]=1
        #Q[3][3]=10000000
        Q[0][0]=1e8
        Q[2][2]=1e8
        R=1*np.identity(B.shape[1])


        x0=x0
        diffModelsExpectation=AircraftLanding.getPertExpectation(T)
        diffModelsVariance=AircraftLanding.getPertVariance(T)

        return System(A,B,C,Q,R,diffModelsExpectation,diffModelsVariance,x0)


    def getPertExpectation2(T):
        '''
        diffExp_i expected perturbations
        '''

        if True:
            diffExp_7=[np.array([[random.uniform(0,0.05)],[random.uniform(0,0.05)]]) for i in range(T)]

            diffExp_6=[np.array([[diffExp_7[i][0][0]+random.uniform(0,0.1)],[diffExp_7[i][1][0]+random.uniform(0,0.5)]]) for i in range(T)]

            diffExp_5=[np.array([[diffExp_6[i][0][0]+random.uniform(0,1)],[diffExp_6[i][1][0]+random.uniform(0,0.1)]]) for i in range(T)]

            diffExp_4=[np.array([[diffExp_5[i][0][0]+random.uniform(0,4)],[diffExp_5[i][1][0]+random.uniform(0,0.5)]]) for i in range(T)]

            diffExp_3=[np.array([[diffExp_4[i][0][0]+random.uniform(0,8)],[diffExp_4[i][1][0]+random.uniform(0,0.3)]]) for i in range(T)]

            diffExp_2=[np.array([[diffExp_3[i][0][0]+random.uniform(0,12)],[diffExp_3[i][1][0]+random.uniform(0,0.5)]]) for i in range(T)]

            diffExp_1=[np.array([[diffExp_2[i][0][0]+random.uniform(0,16)],[diffExp_2[i][1][0]+random.uniform(0,0.7)]]) for i in range(T)]

            diffExp_0=[np.array([[diffExp_1[i][0][0]+random.uniform(0,16)],[diffExp_1[i][1][0]+random.uniform(0,0.7)]]) for i in range(T)]
        else:
            diffExp_7=[np.array([[0],[0]]) for i in range(T)]

            diffExp_6=[np.array([[0],[0]]) for i in range(T)]

            diffExp_5=[np.array([[0],[0]]) for i in range(T)]

            diffExp_4=[np.array([[0],[0]]) for i in range(T)]

            diffExp_3=[np.array([[0],[0]]) for i in range(T)]

            diffExp_2=[np.array([[0],[0]]) for i in range(T)]

            diffExp_1=[np.array([[0],[0]]) for i in range(T)]

            diffExp_0=[np.array([[0],[0]]) for i in range(T)]

        return [diffExp_0,diffExp_1,diffExp_2,diffExp_3,diffExp_4,diffExp_5,diffExp_6,diffExp_7]


    def getPertVariance2(T):
        '''
        diffVar_i perturbation variance
        '''

        if True:

            diffVar_7=[np.array([[random.uniform(0,0.3)],[random.uniform(0,0.05)]]) for i in range(T)]

            diffVar_6=[np.array([[diffVar_7[i][0][0]+random.uniform(0,0.06)],[diffVar_7[i][1][0]+random.uniform(0,0.02)]]) for i in range(T)]

            diffVar_5=[np.array([[diffVar_6[i][0][0]+random.uniform(0,0.08)],[diffVar_6[i][1][0]+random.uniform(0,0.05)]]) for i in range(T)]

            diffVar_4=[np.array([[diffVar_5[i][0][0]+random.uniform(0,0.01)],[diffVar_5[i][1][0]+random.uniform(0,0.02)]]) for i in range(T)]

            diffVar_3=[np.array([[diffVar_4[i][0][0]+random.uniform(0,0.02)],[diffVar_4[i][1][0]+random.uniform(0,0.01)]]) for i in range(T)]

            diffVar_2=[np.array([[diffVar_3[i][0][0]+random.uniform(0,0.02)],[diffVar_3[i][1][0]+random.uniform(0,0.02)]]) for i in range(T)]

            diffVar_1=[np.array([[diffVar_2[i][0][0]+random.uniform(0,0.03)],[diffVar_2[i][1][0]+random.uniform(0,0.03)]]) for i in range(T)]

            diffVar_0=[np.array([[diffVar_1[i][0][0]+random.uniform(0,0.03)],[diffVar_1[i][1][0]+random.uniform(0,0.03)]]) for i in range(T)]
        else:
            diffVar_7=[np.array([[0],[0]]) for i in range(T)]

            diffVar_6=[np.array([[0],[0]]) for i in range(T)]

            diffVar_5=[np.array([[0],[0]]) for i in range(T)]

            diffVar_4=[np.array([[0],[0]]) for i in range(T)]

            diffVar_3=[np.array([[0],[0]]) for i in range(T)]

            diffVar_2=[np.array([[0],[0]]) for i in range(T)]

            diffVar_1=[np.array([[0],[0]]) for i in range(T)]

            diffVar_0=[np.array([[0],[0]]) for i in range(T)]

        return [diffVar_0,diffVar_1,diffVar_2,diffVar_3,diffVar_4,diffVar_5,diffVar_6,diffVar_7]


    def getPertExpectation3(T):
        '''
        diffExp_i expected perturbations
        '''

        if True:
            diffExp_7=[np.array([[random.uniform(0,0.001)],[random.uniform(0,0.001)]]) for i in range(T)]

            diffExp_6=[np.array([[diffExp_7[i][0][0]+random.uniform(0,0.002)],[diffExp_7[i][1][0]+random.uniform(0,0.002)]]) for i in range(T)]

            diffExp_5=[np.array([[diffExp_6[i][0][0]+random.uniform(0,0.01)],[diffExp_6[i][1][0]+random.uniform(0,0.001)]]) for i in range(T)]

            diffExp_4=[np.array([[diffExp_5[i][0][0]+random.uniform(0,0.04)],[diffExp_5[i][1][0]+random.uniform(0,0.005)]]) for i in range(T)]

            diffExp_3=[np.array([[diffExp_4[i][0][0]+random.uniform(0,0.08)],[diffExp_4[i][1][0]+random.uniform(0,0.003)]]) for i in range(T)]

            diffExp_2=[np.array([[diffExp_3[i][0][0]+random.uniform(0,0.12)],[diffExp_3[i][1][0]+random.uniform(0,0.005)]]) for i in range(T)]

            diffExp_1=[np.array([[diffExp_2[i][0][0]+random.uniform(0,0.16)],[diffExp_2[i][1][0]+random.uniform(0,0.007)]]) for i in range(T)]
        else:
            diffExp_7=[np.array([[0],[0]]) for i in range(T)]

            diffExp_6=[np.array([[0],[0]]) for i in range(T)]

            diffExp_5=[np.array([[0],[0]]) for i in range(T)]

            diffExp_4=[np.array([[0],[0]]) for i in range(T)]

            diffExp_3=[np.array([[0],[0]]) for i in range(T)]

            diffExp_2=[np.array([[0],[0]]) for i in range(T)]

            diffExp_1=[np.array([[0],[0]]) for i in range(T)]

        return [diffExp_1,diffExp_2,diffExp_3,diffExp_4,diffExp_5,diffExp_6,diffExp_7]

    def getPertVariance3(T):
        '''
        diffVar_i perturbation variance
        '''

        if True:

            diffVar_7=[np.array([[random.uniform(0,0.0003)],[random.uniform(0,0.0001)]]) for i in range(T)]

            diffVar_6=[np.array([[diffVar_7[i][0][0]+random.uniform(0,0.006)],[diffVar_7[i][1][0]+random.uniform(0,0.002)]]) for i in range(T)]

            diffVar_5=[np.array([[diffVar_6[i][0][0]+random.uniform(0,0.008)],[diffVar_6[i][1][0]+random.uniform(0,0.005)]]) for i in range(T)]

            diffVar_4=[np.array([[diffVar_5[i][0][0]+random.uniform(0,0.001)],[diffVar_5[i][1][0]+random.uniform(0,0.002)]]) for i in range(T)]

            diffVar_3=[np.array([[diffVar_4[i][0][0]+random.uniform(0,0.002)],[diffVar_4[i][1][0]+random.uniform(0,0.001)]]) for i in range(T)]

            diffVar_2=[np.array([[diffVar_3[i][0][0]+random.uniform(0,0.002)],[diffVar_3[i][1][0]+random.uniform(0,0.002)]]) for i in range(T)]

            diffVar_1=[np.array([[diffVar_2[i][0][0]+random.uniform(0,0.003)],[diffVar_2[i][1][0]+random.uniform(0,0.003)]]) for i in range(T)]
        else:
            diffVar_7=[np.array([[0],[0]]) for i in range(T)]

            diffVar_6=[np.array([[0],[0]]) for i in range(T)]

            diffVar_5=[np.array([[0],[0]]) for i in range(T)]

            diffVar_4=[np.array([[0],[0]]) for i in range(T)]

            diffVar_3=[np.array([[0],[0]]) for i in range(T)]

            diffVar_2=[np.array([[0],[0]]) for i in range(T)]

            diffVar_1=[np.array([[0],[0]]) for i in range(T)]

        return [diffVar_1,diffVar_2,diffVar_3,diffVar_4,diffVar_5,diffVar_6,diffVar_7]


    def getPertExpectation(T):
        expLst=MeanVarModels.getDistroAll(T)[0]
        if len(expLst[0])<T:
            print("[Error] ",len(expLst[0]))
        return expLst

    def getPertVariance(T):
        varLst=MeanVarModels.getDistroAll(T)[1]
        if len(varLst[0])<T:
            print("[Error] ",len(varLst[0]))
        return varLst

    def getOracleInstances(T):
        oracleLst=MeanVarModels.getDistroAll(T)[3]
        return oracleLst

    def getModelInstances(T):
        oracleLst=MeanVarModels.getDistroAll(T)[2]
        return oracleLst

if False:
    T=150
    AircraftLanding.getPertExpectation(T)
