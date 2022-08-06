'''
Parameters required for the code.
'''
import os,sys

'''
Please add the following line in ~/.bashrc
export CLD_OFLD_NO_IID_ROOT_DIR = <YOUR PROJECT ROOT>
'''

PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

LIB_PATH=PROJECT_ROOT+'/'+'lib/'
SRC_PATH=PROJECT_ROOT+'/'+'src/'
OUTPUT_PATH=PROJECT_ROOT+'/'+'output/'
PICKLE_PATH=PROJECT_ROOT+'/'+'pickles/'
DATA_PATH=PROJECT_ROOT+'/'+'data/'
MODEL_PATH=PROJECT_ROOT+'/'+'models/'

PICKLE_FLAG=True

#COST=0.000085
#COST=1e10

COST=1e10
#COST=0

MDL_ERR=0.001
ACC_DIFF=0.001


#ALPHA=10000
#BETA=1

#ALPHA=0.07
#BETA=1

ALPHA=0.6
BETA=0.4

EPOCH=200
BATCH=32
DEVICE='cuda'
MODEL_NAME='50'
