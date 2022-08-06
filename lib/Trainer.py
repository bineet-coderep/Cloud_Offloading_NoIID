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

    def trainModel(self,dnnName="Test"):
        '''
        Train a model
        '''

        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='train')
        taxinet_dataset = TensorDataset(all_image_tensor,target_tensor) # create your datset
        loader_train=torch.utils.data.DataLoader(taxinet_dataset, batch_size=self.batch, shuffle=True)
        self.model=OffNet()
        self.model.to(DEVICE)
        ls=torch.nn.MSELoss()
        opt=torch.optim.Adam(self.model.parameters())
        train_loss=[]
        time_taken=time.time()
        for e in range(self.epoch):
            b_tt=time.time()
            for i, (ip,op) in enumerate(loader_train):
                #ip.float().to('cpu')
                #op.float().to('cpu')
                ip.to(DEVICE)
                op.to(DEVICE)
                opt.zero_grad()
                yhat = self.model(ip)
                loss = ls(yhat, op)
                l_val=loss.item()
                loss.backward()
                opt.step()
            train_loss.append(l_val)
            b_tt=time.time()-b_tt
            print(str(e+1)+"/"+str(self.epoch)+";\t Loss: "+str(l_val)+";\t Time Taken: "+str(b_tt)+".")
        time_taken=time.time()-time_taken
        print("========= Training Completed =========")
        print("Time Taken: ",time_taken)
        if self.saveInfoFlag:
            OffModTrainer.plotModelInfo(train_loss)
            torch.save(self.model,DATA_PATH+'/train/'+dnnName+'.ckpt')



    def plotModelInfo(train_loss):
        '''
        Plots training information
        '''
        plt.plot(train_loss)
        plt.title('Training Loss')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.savefig(DATA_PATH+"/train/"+"DNN_Training_Loss", dpi=100, bbox_inches='tight')
        plt.close()

    def test(dnnName="Test"):
        '''
        Test the DNN
        '''
        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='test')
        model=torch.load(DATA_PATH+'/train/'+dnnName+".ckpt")
        model.to(DEVICE)
        all_image_tensor.to(DEVICE)
        target_tensor.to(DEVICE)

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
        print("Model Parameters: ",sum(p.numel() for p in model.parameters()))

    def vizTest(dnnName="Test"):

        (all_image_tensor,target_tensor)=OffModTrainer.dataLoader(type='test')
        model=torch.load(DATA_PATH+'/train/'+dnnName+".ckpt")
        model.to(DEVICE)
        all_image_tensor.to(DEVICE)
        target_tensor.to(DEVICE)
        pred=model(all_image_tensor)
        mse=torch.nn.MSELoss()

        # create a temp dir to visualize a few images
        visualization_dir = DATA_PATH+'/test/'+'/viz/'

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



class OffNet(torch.nn.Module):
    '''
    Define the OffNet
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(44944, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        #print(x.shape)
        #exit(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if False:
    #OffModTrainer.dataLoaderTrain()
    #OffModTrainer().trainModel()
    #OffModTrainer.test()
    OffModTrainer.vizTest()
