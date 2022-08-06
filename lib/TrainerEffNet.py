'''
Provides APIs to train CV models
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *
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




class OffModTrainer:
    '''
    Trains a Model for Offloading
    '''
    def __init__(self,batch=BATCH,epoch=EPOCH,saveInfoFlag=True):
        self.batch=batch
        self.epoch=epoch
        self.saveInfoFlag=True

    def dataLoader(type='train'):
        labelDataFname=DATA_PATH+'/'+type+'/'
        torch.cuda.empty_cache()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print('found device: ', device)

        # where the training results should go
        #results_dir = remove_and_create_dir(SCRATCH_DIR + '/DNN_train_taxinet/')

        # where raw images and csvs are saved
        #BASE_DATALOADER_DIR = DATA_DIR + '/' + dataset_type  + '/' + condition

        # how often to plot a few images for progress report
        # warning: plotting is slow
        NUM_PRINT = 2

        IMAGE_WIDTH = 224
        IMAGE_HEIGHT = 224

        # create a temp dir to visualize a few images
        #visualization_dir = SCRATCH_DIR + '/viz/'
        #remove_and_create_dir(visualization_dir)

        # where the final dataloader will be saved
        #DATALOADER_DIR = remove_and_create_dir(SCRATCH_DIR + '/dataloader/')

        MAX_FILES = np.inf

        # where original XPLANE images are stored
        #data_dir = DATA_DIR + '/test_dataset_smaller_ims/'


        # resize to 224 x 224 x 3 for EfficientNets
        # prepare image transforms
        # warning: you might need to change the normalization values given your dataset's statistics
        tfms = transforms.Compose([transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]),])


        image_list = [x for x in os.listdir(labelDataFname+'/images/') if x.endswith('.png')]

        # dataframe of labels
        labels_df = pandas.read_csv(labelDataFname+'altitude.csv', sep=',')


        # loop through images and save in a dataloader
        image_tensor_list = []

        # tensor of targets y:  modify to whatever you want to predict from DNN
        target_tensor_list = []

        for i, image_name in enumerate(image_list):

            # open images and apply transforms
            fname = labelDataFname + '/images/' + str(image_name)
            image = Image.open(fname).convert('RGB')
            tensor_image_example = tfms(image)

            # add image
            image_tensor_list.append(tensor_image_example)

            # get the corresponding state information (labels) for each image
            specific_row = labels_df[labels_df['image_filename'] == image_name]
            # there are many states of interest, you can modify to access which ones you want
            h = specific_row['h'].item()
            #theta = specific_row['pitch'].item()
            # add tensor
            #target_tensor_list.append([dist_centerline_norm, downtrack_position_norm, heading_error_norm])
            target_tensor_list.append([h])


        # first, save image tensors
        # concatenate all image tensors
        all_image_tensor = torch.stack(image_tensor_list)
        #print(all_image_tensor.shape)

        # save tensors to disk
        image_data = labelDataFname + '/images_airsim.pt'
        # sizes are: 126 images, 3 channels, 224 x 224 each
        # torch.Size([126, 3, 224, 224])
        torch.save(all_image_tensor, image_data)

        ###################################
        # second, save target label tensors
        target_tensor = torch.tensor(target_tensor_list)
        #print(target_tensor.shape)

        # size: 126 numbers by 3 targets
        # torch.Size([126])

        # save tensors to disk
        target_data = labelDataFname + '/target_airsim.pt'
        torch.save(target_tensor, target_data)

        # all_image_tensor
        # target_tensor

        #print(target_tensor[:20])

        return (all_image_tensor,target_tensor)

    def freezeModel(model):
        # freeze everything
        n_params = len(list(model.parameters()))
        for i, p in enumerate(model.parameters()):
            p.requires_grad = False

        # make last layer trainable
        for p in model._fc.parameters():
            p.requires_grad = True

        return model

    def trainModel(self,outDim=1,modelname=MODEL_NAME):
        '''
        Train a model
        '''
        print(">> STATUS: Data Loading Started!!")
        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='train')
        print(">> STATUS: Data Loaded!!")
        print(">> STATUS: Started Tensoring The Data!!")
        taxinet_dataset = TensorDataset(all_image_tensor,target_tensor) # create your datset
        loader_train=torch.utils.data.DataLoader(taxinet_dataset, batch_size=self.batch, shuffle=True)
        print(">> STATUS: Data Tensored!!")
        print(">> STATUS: Freezing Model!!")
        self.model=EfficientNet.from_pretrained('efficientnet-b'+modelname,num_classes=outDim)
        #self.model._fc = torch.nn.Linear(1280, outDim)
        self.model=OffModTrainer.freezeModel(self.model)
        print(">> STATUS: Model Frozen!!")
        print(">> STATUS: Preparing Model For Training!!")
        self.model.train()
        self.model=self.model.to(DEVICE)
        ls=torch.nn.MSELoss()
        opt=torch.optim.Adam(self.model.parameters())
        train_loss=[]
        print(">> STATUS: Model Ready To be Trained!!")
        time_taken=time.time()
        #print("B")
        print(">> STATUS: Model Training Started!!")
        for e in range(self.epoch):
            b_tt=time.time()
            #print("\tC")
            for i, (ip,op) in enumerate(loader_train):
                sys.stdout.write('\r')
                sys.stdout.write("\tEpoch: "+str(e+1)+";\tBatch: "+str(i+1)+"/"+str(len(loader_train)))
                sys.stdout.flush()
                ip=ip.float().to(DEVICE)
                op=op.float().to(DEVICE)
                #print("\t\tD")
                #inputs, labels = inputs.to(device)
                #labels.to(device)
                #print("\t\tD2")
                #print("\t\tD3")
                opt.zero_grad()
                #print("\t\tD4")
                yhat = self.model(ip)
                #print("\t\tD5")
                loss = ls(yhat, op)
                #print("\t\tD6")
                l_val=loss.item()
                #print("\t\tD7")
                loss.backward()
                #print("\t\tD8")
                opt.step()
                #print("\t\tD9")
            train_loss.append(l_val)
            b_tt=time.time()-b_tt
            print("\n"+str(e+1)+"/"+str(self.epoch)+";\t Loss: "+str(l_val)+";\t Time Taken: "+str(b_tt)+".")
        time_taken=time.time()-time_taken
        print(">> STATUS: Model Trained!!")
        print("========= Training Completed =========")
        print("Time Taken: ",time_taken)
        torch.cuda.empty_cache()
        if self.saveInfoFlag:
            OffModTrainer.plotModelInfo(train_loss,modelname)
            torch.save(self.model,MODEL_PATH+'/'+'efficientnet-b'+modelname+'.ckpt')

    def plotModelInfo(train_loss,modelname=MODEL_NAME):
        '''
        Plots training information
        '''
        plt.plot(train_loss)
        plt.title('Training Loss')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig(MODEL_PATH+'/'+"DNN_Training_Loss_efficientnet-b"+modelname, dpi=100, bbox_inches='tight')
        plt.close()

    def test2(modelname=MODEL_NAME):
        '''
        Test the DNN
        '''
        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='test')
        model=torch.load(MODEL_PATH+'/'+'efficientnet-b'+modelname+".ckpt")
        model=model.to('cpu')
        model.eval()
        #all_image_tensor=all_image_tensor[:10]
        #target_tensor=target_tensor[:10]
        all_image_tensor=all_image_tensor.to('cpu')
        target_tensor=target_tensor.to('cpu')
        #ip.float().to(DEVICE)
        #op.float().to(DEVICE)

        time_taken=time.time()
        predictions = model(all_image_tensor)
        predictions.to(DEVICE)
        print("# Model")
        mae = torch.nn.L1Loss()
        print("MAE: ",mae(target_tensor, predictions).item())
        mse = torch.nn.MSELoss()
        print("MSE: ",mse(target_tensor, predictions).item())
        time_taken=time.time()-time_taken
        print("Total Time Taken: ",time_taken)
        print("Average Time Taken: ",time_taken/len(all_image_tensor))
        mPar=sum(p.numel() for p in model.parameters())
        print("Model Parameters: ",mPar)

        return (mae,mse,mPar)

    def test(modelname=MODEL_NAME,batch_size=BATCH):
        '''
        Test the DNN
        '''

        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='test')
        al_dataset = TensorDataset(all_image_tensor,target_tensor) # create your datset
        loader_train=torch.utils.data.DataLoader(al_dataset, batch_size=batch_size, shuffle=False)
        model=torch.load(MODEL_PATH+'/'+'efficientnet-b'+modelname+".ckpt")
        model=model.to(DEVICE)
        model.eval()

        lsLstMSE=[]
        lsLstMAE=[]
        mse=torch.nn.MSELoss()
        mae = torch.nn.L1Loss()
        time_taken=time.time()
        for i, (ip,op) in enumerate(loader_train):
            ip=ip.float().to(DEVICE)
            op=op.float().to(DEVICE)
            yhat = model(ip)
            lossMSE = mse(yhat, op)
            lossMAE = mae(yhat, op)
            lsLstMSE.append(lossMSE.item())
            lsLstMAE.append(lossMAE.item())
        time_taken=time.time()-time_taken


        print("# Model")
        maeLss=sum(lsLstMAE)/len(all_image_tensor)
        print("MAE: ",maeLss)
        mseLss=sum(lsLstMSE)/len(all_image_tensor)
        print("MSE: ",mseLss)
        print("Total Time Taken: ",time_taken)
        print("Average Time Taken: ",time_taken/len(all_image_tensor))
        mPar=sum(p.numel() for p in model.parameters())
        print("Model Parameters: ",mPar)

        return (maeLss,mseLss,mPar,lsLstMAE,lsLstMSE)

    def vizTest(modelname=MODEL_NAME):

        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='test')
        #print(all_image_tensor.shape)
        #all_image_tensor=all_image_tensor[:10]
        #target_tensor=target_tensor[:10]
        model=torch.load(MODEL_PATH+'/'+'efficientnet-b'+modelname+".ckpt")
        model=model.to('cpu')
        all_image_tensor=all_image_tensor.to('cpu')
        target_tensor=target_tensor.to('cpu')
        pred=model(all_image_tensor)
        mse=torch.nn.MSELoss()

        # create a temp dir to visualize a few images
        visualization_dir = MODEL_PATH+'/viz/'

        MAX_LIMIT=20
        for i in range(len(all_image_tensor)):
            A=all_image_tensor[i]
            torchvision.utils.save_image(A,visualization_dir+'/'+'img'+str(i)+'.png')
            A_img = cv2.imread(visualization_dir+'/'+'img'+str(i)+'.png')
            ls=mse(target_tensor[i],pred[i]).item()
            h_pred=pred[i][0].tolist()
            h_oracle=target_tensor[i][0].tolist()
            title_str="Oracle: "+format(h_oracle,'.5f')+". "+"Test: "+format(h_pred,'.5f')
            #print(title_str)
            plt.imshow(A_img)
            plt.title(title_str)
            plt.savefig(visualization_dir+'/'+'img'+str(i)+'.png')
            if i>MAX_LIMIT:
                break
            #plt.close()
            #cv2.imwrite('myImage.png',A)

    def trainAll(self):
        for m in range(8):
            print("=== TRAINING MODEL: EfficientNet-"+str(m)+"\n")
            self.trainModel(modelname=str(m))
            print("\n==========================\n\n")
            if batch_size>2:
                batch_size=batch_size/2

    def testAll():
        lssLstMSE=[]
        lssLstMAE=[]
        mse_lst=[]
        mae_lst=[]
        prmLst=[]
        #batch_size=BATCH
        for m in range(8):
            print("=== TESTING MODEL: EfficientNet-"+str(m)+"\n")
            (mae,mse,mPar,lstMAE,lstMSE)=OffModTrainer.test(modelname=str(m))
            lssLstMSE.append(mse)
            lssLstMAE.append(mae)
            prmLst.append(mPar)
            mse_lst.append([(mPar,mse)])
            mse_lst.append([(mPar,mae)])
            print("\n==========================\n\n")
        mse_lst_df = pd.DataFrame(mse_lst,columns=['par','mse'])
        mae_lst_df = pd.DataFrame(mae_lst,columns=['par','mae'])
        OffModTrainer.plotEffNetModel2(mse_lst_df)
        OffModTrainer.plotEffNetModel(lssLstMSE,lssLstMAE,prmLst)

    def plotEffNetModel2(mse_lst_df):
        plt.title('EfficientNet Model (Parameters vs. Loss)')
        #plt.scatter(hList,diffList,alpha=0.1,color='cyan')
        #sns.lineplot(x="sepal_length", y="sepal_width", data=data)
        p = sns.lineplot(x="par", y="mse", data=mse_lst_df)
        p.set_xlabel('MSE', fontsize=11,fontweight='bold')
        p.set_ylabel('Parameters', fontsize=11,fontweight='bold')
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+"EfficientNetModels_MSE"+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)
        plt.close()

    def plotEffNetModel(lssLstMSE,lssLstMAE,prmLst):
        '''
        Plots training information
        '''
        #plt.plot(prmLst,lssLstMSE)
        plt.plot(prmLst, lssLstMSE, 'go--', linewidth=2, markersize=12)
        plt.plot(prmLst, lssLstMAE, 'bo--', linewidth=2, markersize=12)
        plt.title('EfficientNet Model (Parameters vs. Loss)')
        plt.ylabel('MSE')
        plt.xlabel('Parameters')
        plt.show()
        plt.savefig(MODEL_PATH+'/'+"EfficientNetModels", dpi=100, bbox_inches='tight')
        plt.close()


def trainAllBatch():

    batchSizes=[30,20,18,14,10,6,4,2]
    #epochList=[40,50,60,70,80,90,100,110]
    epochList=[20]*8
    for m in range(8):
        print("=== TRAINING MODEL: EfficientNet-"+str(m)+"\n")
        OffModTrainer(batch=batchSizes[m],epoch=epochList[m]).trainModel(modelname=str(m))
        print("\n==========================\n\n")


if False:
    #OffModTrainer.dataLoaderTrain()
    #OffModTrainer().trainModel()
    #OffModTrainer.test(modelname="0")
    #print("\n==============\n")
    #OffModTrainer.test2(modelname="0")
    #OffModTrainer.vizTest()
    #OffModTrainer().trainAll()
    #OffModTrainer.testAll()
    trainAllBatch()
