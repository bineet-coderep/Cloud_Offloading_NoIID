'''
Provides an example Quadrotor System.
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *
from lib.System import *

import numpy as np
import random


class QuadSystem:

    def getSimpleSystem(x0,T):
        '''
        Define the dynamics matrices
        '''
        A2=np.array([
        [0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,-9.8,0,0,0,0],
        [0,0,0,0,0,0,9.8,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0]
        ])
        B2=np.array([
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [1,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ])
        h=0.01
        A=np.add(np.identity(A2.shape[0]),h*A2)
        B=h*B2
        C=np.array([
        [1,0],
        [0,1],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0]
        ])

        Q=1*np.identity(A.shape[0])
        R=1*np.identity(B.shape[1])

        x0=np.array([x0]).T
        diffModelsExpectation=QuadSystem.getPertExpectation(T)
        diffModelsVariance=QuadSystem.getPertVariance(T)

        return System(A,B,C,Q,R,diffModelsExpectation,diffModelsVariance,x0)


    def getPertExpectation(T):
        '''
        diffExp_i expected perturbations
        '''

        if True:
            diffExp_7=[np.array([[random.uniform(0,0.05)],[random.uniform(0,0.05)]]) for i in range(T)]

            diffExp_6=[np.array([[diffExp_7[i][0][0]+random.uniform(0,0.12)],[diffExp_7[i][1][0]+random.uniform(0,0.5)]]) for i in range(T)]

            diffExp_5=[np.array([[diffExp_6[i][0][0]+random.uniform(0,0.15)],[diffExp_6[i][1][0]+random.uniform(0,0.1)]]) for i in range(T)]

            diffExp_4=[np.array([[diffExp_5[i][0][0]+random.uniform(0,0.1)],[diffExp_5[i][1][0]+random.uniform(0,0.1)]]) for i in range(T)]

            diffExp_3=[np.array([[diffExp_4[i][0][0]+random.uniform(0,0.4)],[diffExp_4[i][1][0]+random.uniform(0,0.3)]]) for i in range(T)]

            diffExp_2=[np.array([[diffExp_3[i][0][0]+random.uniform(0,0.5)],[diffExp_3[i][1][0]+random.uniform(0,0.5)]]) for i in range(T)]

            diffExp_1=[np.array([[diffExp_2[i][0][0]+random.uniform(0,0.6)],[diffExp_2[i][1][0]+random.uniform(0,0.7)]]) for i in range(T)]
        else:
            diffExp_7=[np.array([[0],[0]]) for i in range(T)]

            diffExp_6=[np.array([[0],[0]]) for i in range(T)]

            diffExp_5=[np.array([[0],[0]]) for i in range(T)]

            diffExp_4=[np.array([[0],[0]]) for i in range(T)]

            diffExp_3=[np.array([[0],[0]]) for i in range(T)]

            diffExp_2=[np.array([[0],[0]]) for i in range(T)]

            diffExp_1=[np.array([[0],[0]]) for i in range(T)]

        return [diffExp_1,diffExp_2,diffExp_3,diffExp_4,diffExp_5,diffExp_6,diffExp_7]


    def getPertVariance(T):
        '''
        diffVar_i perturbation variance
        '''

        diffVar_7=[np.array([[random.uniform(0,0.3)],[random.uniform(0,0.05)]]) for i in range(T)]

        diffVar_6=[np.array([[diffVar_7[i][0][0]+random.uniform(0,0.06)],[diffVar_7[i][1][0]+random.uniform(0,0.02)]]) for i in range(T)]

        diffVar_5=[np.array([[diffVar_6[i][0][0]+random.uniform(0,0.08)],[diffVar_6[i][1][0]+random.uniform(0,0.05)]]) for i in range(T)]

        diffVar_4=[np.array([[diffVar_5[i][0][0]+random.uniform(0,0.01)],[diffVar_5[i][1][0]+random.uniform(0,0.02)]]) for i in range(T)]

        diffVar_3=[np.array([[diffVar_4[i][0][0]+random.uniform(0,0.02)],[diffVar_4[i][1][0]+random.uniform(0,0.01)]]) for i in range(T)]

        diffVar_2=[np.array([[diffVar_3[i][0][0]+random.uniform(0,0.02)],[diffVar_3[i][1][0]+random.uniform(0,0.02)]]) for i in range(T)]

        diffVar_1=[np.array([[diffVar_2[i][0][0]+random.uniform(0,0.03)],[diffVar_2[i][1][0]+random.uniform(0,0.03)]]) for i in range(T)]

        return [diffVar_1,diffVar_2,diffVar_3,diffVar_4,diffVar_5,diffVar_6,diffVar_7]
