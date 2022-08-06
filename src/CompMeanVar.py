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

from Parameters import *

from lib.MeanVar import *

if True:
    T=150
    MeanVarModels.getDistroAll(T)
