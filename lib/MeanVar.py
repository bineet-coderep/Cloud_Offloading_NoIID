'''
Provides APIs to compute mean and variance
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *
from lib.TrainerEffNetEarlyStop import *
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pandas
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import pickle
import os
import torchvision
from efficientnet_pytorch import EfficientNet
import sys
import statistics as stat
import copy
import tensorflow as tf
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class MeanVarModels:

    def getDistroModel(modelname=MODEL_NAME):
        '''
        Test the DNN
        '''

        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='test')
        al_dataset = TensorDataset(all_image_tensor,target_tensor) # create your datset
        loader_train=torch.utils.data.DataLoader(al_dataset, batch_size=1, shuffle=False)
        model=torch.load(MODEL_PATH+'/'+'efficientnet-b'+modelname+".ckpt")
        model=model.to(DEVICE)
        model.eval()

        lsLstMSE=[]
        lsLstMAE=[]
        mse=torch.nn.MSELoss()
        mae = torch.nn.L1Loss()
        time_taken=time.time()

        hList=[]
        diffList=[]
        oracleList=[]
        yhatList=[]

        for i, (ip,op) in enumerate(loader_train):
            #print("h (oracle): ", op[0][0].tolist())
            hList.append(op[0][0].tolist())
            ip=ip.float().to(DEVICE)
            op=op.float().to(DEVICE)
            yhat = model(ip)
            #print("h (pred): ",yhat[0][0].tolist())
            diffList.append(yhat[0][0].tolist()-op[0][0].tolist())
            oracleList.append([[op[0][0].tolist()]])
            yhatList.append([[yhat[0][0].tolist()]])
            #print("\n")
            yhat=yhat.to(DEVICE)
            lossMSE = mse(yhat, op)
            lossMAE = mae(yhat, op)
            lsLstMSE.append(lossMSE.item())
            lsLstMAE.append(lossMAE.item())
        time_taken=time.time()-time_taken

        return (hList,diffList)


    def getDistroAll(nSteps):

        if os.path.exists(MODEL_PATH+'/model_stats_'+str(nSteps)+'.pkl'):
            pkl_file = open(MODEL_PATH+'/model_stats_'+str(nSteps)+'.pkl', 'rb')
            pklDS = pickle.load(pkl_file)
            pkl_file.close()
            modelExpLst=pklDS['expectation']
            modelVarLst=pklDS['variance']

            return (modelExpLst,modelVarLst)

        modelVarLst=[]
        modelExpLst=[]

        for m in range(8):

            meanLst=[]
            varLst=[]

            print("======== EfficientNet ",str(m),"========")
            (hList,diffList)=MeanVarModels.getDistroModel(str(m))
            maxH=max(hList)
            #print(maxH,nSteps)
            stepSize=maxH/nSteps
            #print(stepSize)

            blockH=np.zeros(nSteps,dtype=object)

            for i in range(nSteps):
                blockH[i]=[]

            mean_var_list=[]
            for i in range(nSteps):
                lb=i*stepSize
                ub=(i+1)*stepSize
                for j in range(len(hList)):
                    if hList[j]>=lb and hList[j]<ub:
                        blockH[i].append(diffList[j])
                        mean_var_list.append([(lb+ub)/2,diffList[j]])

            if False:
                trialsList=[]
                nTrials=np.inf
                for i in range(nSteps):
                    print(len(blockH[i]))
                    if nTrials>len(blockH[i]):
                        nTrials=len(blockH[i])
                print(nTrials)
                for i in range(nTrials):
                    trialList=[]
                    for j in range(nSteps):
                        trialList.append(blockH[j][i])
                    trialsList.append(trialList)
                exit(0)

            if True:
                meanList=[]
                varList=[]
                rangeList=[]
                rangeMPList=[]
                for i in range(nSteps):
                    lb=i*stepSize
                    ub=(i+1)*stepSize
                    if len(blockH[i])>1:
                        meanList.append([stat.mean(blockH[i])])
                        varList.append([stat.stdev(blockH[i])])
                        rangeMPList.append((lb+ub)/2)
                        rangeList.append((lb,ub))
                    else:
                        0;
                        meanList.append([0])
                        varList.append([0])

            modelExpLst.append(meanList)
            modelVarLst.append(varList)
            mean_var_df = pd.DataFrame(mean_var_list,columns=['h','diff'])

            MeanVarModels.plotDistro(hList,diffList,mean_var_df,fname=str(m))
        pklDS={
        'expectation': modelExpLst,
        'variance': modelVarLst
        }
        output = open(MODEL_PATH+'/model_stats_'+str(nSteps)+'.pkl', 'wb')
        pickle.dump(pklDS, output)
        output.close()

        return (modelExpLst,modelVarLst)





    def plotDistro(hList,diffList,mean_var_df,fname):
        '''
        Plots training information
        '''

        #print(mean_var_df)
        # loading dataset
        #data = sns.load_dataset("iris")
        #print(data)

        #plt.ylabel('predicted - actual')
        #plt.xlabel('h')
        plt.title('Timewise Distribution of the Model b'+str(fname),fontsize=16,fontweight='bold')
        #plt.scatter(hList,diffList,alpha=0.1,color='cyan')
        #sns.lineplot(x="sepal_length", y="sepal_width", data=data)
        p = sns.lineplot(x="h", y="diff", data=mean_var_df)
        p.set_xlabel(r'$\mathbf{h}$ (altitude)', fontsize=11,fontweight='bold')
        p.set_ylabel("predicted - actual", fontsize=11,fontweight='bold')
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+"model_distro_"+fname+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)
        plt.close()




if False:
    MeanVarModels.getDistroAll(150)
