'''
Provides APIs to sanitize the dataset
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *
import random
import scipy.linalg as LA
import numpy as np
from gurobipy import *
import time
import math

class SanitizeDSet:
    def __init__(self,helipadLoc=(110,32.60,-10),datsetFname="airsim_rec.txt"):
        self.helipadLoc=helipadLoc # (x,y,z)
        self.datasetFname=datsetFname

    def prepData(self,type='train'):
        fname=DATA_PATH+'/'+type+'/'+self.datasetFname
        file1 = open(fname, 'r')
        lines = file1.readlines()
        nwLines=['image_filename,h\n']
        nL=len(lines)
        for i in range(1,nL):
            line=lines[i]
            tmp1=line.split()
            if len(tmp1)>0:
                posX=float(tmp1[2])
                posY=float(tmp1[3])
                posZ=float(tmp1[4])
                h=math.sqrt(((posX-self.helipadLoc[0])**2)+((posY-self.helipadLoc[1])**2)+((posZ-self.helipadLoc[2])**2))
                nwLine=tmp1[-1]+","+str(h)+"\n"
                #print(nwLine)
                nwLines.append(nwLine)

        # writing to file
        fname2=DATA_PATH+'/'+type+'/'+"altitude.csv"
        file2 = open(fname2, 'w')
        file2.writelines(nwLines)
        file2.close()

    def prepData2(self,type='train'):
        fname=DATA_PATH+'/'+type+'/'+self.datasetFname
        file1 = open(fname, 'r')
        lines = file1.readlines()
        nwLines=['image_filename,h,pitch\n']
        nL=len(lines)
        for i in range(1,nL):
            line=lines[i]
            tmp1=line.split()
            if len(tmp1)>0:
                posX=float(tmp1[2])
                posY=float(tmp1[3])
                posZ=float(tmp1[4])
                theta=float(tmp1[5])
                h=math.sqrt(((posX-self.helipadLoc[0])**2)+((posY-self.helipadLoc[1])**2)+((posZ-self.helipadLoc[2])**2))
                nwLine=tmp1[-1]+","+str(h)+","+str(theta)+"\n"
                #print(nwLine)
                nwLines.append(nwLine)

        # writing to file
        fname2=DATA_PATH+'/'+type+'/'+"altitude_pitch.csv"
        file2 = open(fname2, 'w')
        file2.writelines(nwLines)
        file2.close()






if False:
    SanitizeDSet().prepData(type='validation')
