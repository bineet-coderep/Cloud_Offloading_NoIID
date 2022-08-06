'''
Implements various functionalities using /lib/
'''
import os,sys,time
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *

from lib.Offloader import *
from lib.QuadrotorSystem import *
from lib.Visualize import *
from lib.Evaluater import *
#from lib.System import *
import random
import statistics as stat
from scipy.optimize import curve_fit



class ExpQuad:

    def checkToyModel(x0,T):
        toySystem=QuadSystem.getSimpleSystem(x0,T)


        expLosses=[]
        expCosts=[]

        trueLosses=[]
        trueCosts=[]

        timeStep=random.randint(0,T)

        TRIALS=100
        for m in range(1,8):
            exp=toySystem.diffModelsExpectation[m-1][timeStep]
            var=toySystem.diffModelsVariance[m-1][timeStep]
            expLoss=(exp[0][0]*exp[0][0])+(exp[1][0]*exp[1][0]).item()
            expLosses.append(expLoss)
            expCost=COST*m
            expCosts.append(expCost)
            trialLosses=[]
            trialCosts=[]
            for i in range(TRIALS):
                val0=np.random.normal(exp[0][0],np.sqrt(var[0][0]))
                val1=np.random.normal(exp[1][0],np.sqrt(var[1][0]))
                valloss=(val0*val0)+(val1*val1)
                trialLosses.append(valloss)
                trialCosts.append(expCost)
            trueLosses.append(trialLosses)
            trueCosts.append(trialCosts)


        ExpQuad.vizLossCost(trueLosses,trueCosts,expLosses,expCosts)

        timeSteps=[random.randint(0,T-1) for k in range(10)]
        lossesList=[]
        costsList=[]

        for timeStep in timeSteps:

            losses=[]
            costs=[]
            for m in range(1,8):
                exp=toySystem.diffModelsExpectation[m-1][timeStep]
                loss=(exp[0][0]*exp[0][0])+(exp[1][0]*exp[1][0]).item()
                losses.append(loss)
                cost=COST*m
                costs.append(cost)
            lossesList.append(losses)
            costsList.append(costs)

        ExpQuad.vizSimpleLossCost(lossesList,costsList)


    def checkToyModelContinous(x0,T):
        timeStep=random.randint(0,T)
        toySystem=QuadSystem.getSimpleSystem(x0,T)
        expRbt=toySystem.diffModelsExpectation[0][timeStep]
        varRbt=toySystem.diffModelsVariance[0][timeStep]
        expCld=toySystem.diffModelsExpectation[6][timeStep]
        varCld=toySystem.diffModelsVariance[6][timeStep]

        expLosses=[]
        expCosts=[]

        trueLosses=[]
        trueCosts=[]

        STEPS=100
        c=0.01
        TRIALS=100

        for i in range(STEPS):
            exp=((1-c)*expRbt)+(c*expCld)
            var=((1-c)*varRbt)+(c*varCld)
            expLoss=(exp[0][0]*exp[0][0])+(exp[1][0]*exp[1][0]).item()
            expLosses.append(expLoss)
            expCost=COST*c
            expCosts.append(expCost)
            trialLosses=[]
            trialCosts=[]
            for i in range(TRIALS):
                val0=np.random.normal(exp[0][0],np.sqrt(var[0][0]))
                val1=np.random.normal(exp[1][0],np.sqrt(var[1][0]))
                valloss=(val0*val0)+(val1*val1)
                trialLosses.append(valloss)
                trialCosts.append(expCost)
            trueLosses.append(trialLosses)
            trueCosts.append(trialCosts)
            c=c+0.01


        ExpQuad.vizLossCost(trueLosses,trueCosts,expLosses,expCosts)

        timeSteps=[random.randint(0,T-1) for k in range(10)]
        lossesList=[]
        costsList=[]

        for timeStep in timeSteps:

            losses=[]
            costs=[]
            STEPS=100
            c=0.01
            expRbt=toySystem.diffModelsExpectation[0][timeStep]
            expCld=toySystem.diffModelsExpectation[6][timeStep]
            for i in range(STEPS):
                exp=((1-c)*expRbt)+(c*expCld)
                loss=(exp[0][0]*exp[0][0])+(exp[1][0]*exp[1][0]).item()
                losses.append(loss)
                cost=COST*c
                costs.append(cost)
                c=c+0.01
            lossesList.append(losses)
            costsList.append(costs)

        ExpQuad.vizSimpleLossCost(lossesList,costsList)


    def vizSimpleLossCost(lossesList,costsList):
        plt.figure()
        plt.ylabel("Loss")
        plt.xlabel("Cost")


        for costs,losses in zip(costsList,lossesList):
            plt.plot(costs,losses,linewidth=2)
        plt.show()

    def vizLossCost(trueLosses,trueCosts,expLosses,expCosts):
        plt.figure()
        plt.ylabel("Loss")
        plt.xlabel("Cost")

        for loss,cost in zip(trueLosses,trueCosts):
            0;
            #print(len(loss))
            plt.scatter(cost,loss,s=200,alpha=0.1)
            plt.scatter(stat.mean(cost),stat.mean(loss),s=300,color='blue')


        plt.scatter(expCosts,expLosses,s=300,color='k')

        plt.show()

        plt.clf()

        plt.scatter(expCosts,expLosses,s=300,color='k')
        plt.plot(expCosts,expLosses,color='k',linewidth=4)
        plt.show()


    def checkOffloading(x0,T):
        toySystem=QuadSystem.getSimpleSystem(x0,T)
        ofldr=OffloaderExpectation(toySystem)
        optChoices=ofldr.getOptOffloader()
        randChoices=ofldr.getRandOffloader()

        #evl=Evaluater(toySystem)
        #print(evl.getPerCost([7]*T))
        #print(evl.getControlCost([1]*T))



        print(optChoices)

        ExpQuad.vizModelChoices(optChoices)

        #exit(0)

        # Evaluate
        TRIALS=100
        oracleRewards=[]
        allCloudRewards=[]
        allRobotRewards=[]
        randRewards=[]
        optRewards=[]

        oraclePerCost=[]
        allCloudPerCost=[]
        allRobotPerCost=[]
        randPerCost=[]
        optPerCost=[]

        oracleCtrlCost=[]
        allCloudCtrlCost=[]
        allRobotCtrlCost=[]
        randCtrlCost=[]
        optCtrlCost=[]

        for i in range(TRIALS):
            evl=Evaluater(toySystem)
            randChoices=ofldr.getRandOffloader()

            optRewards.append(evl.computeReward(optChoices))
            oracleRewards.append(evl.computeOracleReward())
            allCloudRewards.append(evl.computeReward([7]*T))
            allRobotRewards.append(evl.computeReward([1]*T))
            randRewards.append(evl.computeReward(randChoices))

            optPerCost.append(evl.getPerCost(optChoices))
            oraclePerCost.append(0)
            allCloudPerCost.append(evl.getPerCost([7]*T))
            allRobotPerCost.append(evl.getPerCost([1]*T))
            randPerCost.append(evl.getPerCost(randChoices))

            optCtrlCost.append(evl.getControlCost(optChoices))
            oracleCtrlCost.append(evl.getOracleControlCost())
            allCloudCtrlCost.append(evl.getControlCost([7]*T))
            allRobotCtrlCost.append(evl.getControlCost([1]*T))
            randCtrlCost.append(evl.getControlCost(randChoices))

        print("Mean Rewards: ",stat.mean(optRewards),stat.mean(randRewards))

        labels=["Oracle","Optimal","Random","Cloud(7)","Robot(1)"]
        rewards=[oracleRewards,optRewards,randRewards,allCloudRewards,allRobotRewards]
        perCosts=[oraclePerCost,optPerCost,randPerCost,allCloudPerCost,allRobotPerCost]
        controlCosts=[oracleCtrlCost,optCtrlCost,randCtrlCost,allCloudCtrlCost,allRobotCtrlCost]

        ExpQuad.vizStats(labels,perCosts,controlCosts,rewards)



    def checkOffloadingVaryingAB(x0,T):
        toySystem=QuadSystem.getSimpleSystem(x0,T)
        evl=Evaluater(toySystem)

        ctrlCosts=[]
        perCosts=[]


        alpha=0
        STEP=100

        for i in range(STEP):
            ofldr=OffloaderExpectation(toySystem)
            optChoices=ofldr.getOptOffloader(alpha=alpha,beta=(1-alpha))
            randChoices=ofldr.getRandOffloader()

            perCosts.append(evl.getPerCost(optChoices))
            ctrlCosts.append(evl.getControlCost(optChoices))
            alpha=alpha+0.01

        print(perCosts)
        print(ctrlCosts)

        ExpQuad.vizParetoOpti(ctrlCosts,perCosts)


    def vizParetoOpti(ctrlCosts,perCosts):
        plt.figure()
        plt.ylabel("Control Cost")
        plt.xlabel("Perception Cost")


        plt.plot(perCosts,ctrlCosts,linewidth=2)
        plt.scatter(perCosts[0],ctrlCosts[0],s=100,color='black')
        plt.scatter(perCosts[-1],ctrlCosts[-1],s=100,color='green')
        plt.show()




    def checkOffloadingContinous(x0,T):
        toySystem=QuadSystem.getSimpleSystem(x0,T)
        ofldr=OffloaderExpectation(toySystem)
        optChoices=ofldr.getOptOffloaderContinous(cost=70)

        print(optChoices)

        ExpQuad.vizModelChoices(optChoices)


    def vizModelChoices(modelChoices):
        plt.figure()
        plt.ylabel("Models")
        plt.xlabel("Time")
        X=list(range(len(modelChoices)))


        plt.plot(X,modelChoices,marker='o',markersize=7,linewidth=0.2,color='blue')
        plt.show()

    def vizStats(labels,perCosts,controlCosts,rewards,th0=0,th1=1,fname="simple_system_"):
        plt.figure()

        plt.xlabel("Perception Cost")
        plt.ylabel("Control Cost")
        #plt.yticks(list(range(1080000,111000,100)))

        for (label,perCost,controlCost) in zip(labels,perCosts,controlCosts):
            #print(label)
            meanPerCost=stat.mean(perCost)
            meanCtrlCost=stat.mean(controlCost)
            plt.scatter(perCost,controlCost,s=100,marker='o',alpha=0.1)
            plt.text(meanPerCost,meanCtrlCost,label,horizontalalignment='left',color='k')
            plt.plot(meanPerCost,meanCtrlCost,markersize=15,marker='o')

        #plt.legend()
        plt.show()
        #plt.savefig(OUTPUT_PATH+'/'+fname+"costs")
        plt.close()


        rewards_new=[]
        labels_new=[]
        for i in range(len(labels)):
            for rwd in rewards[i]:
                rewards_new.append(rwd)
                labels_new.append(labels[i])

        rwdDict={
        'rewards':rewards_new,
        'policy':labels_new
        }
        rwds = pd.DataFrame(rwdDict)
        plt.figure()
        ax = sns.boxplot(x="policy", y="rewards", data=rwds)
        plt.ylabel('Rewards',fontweight="bold",fontsize=15)
        plt.xlabel('Policies',fontweight="bold",fontsize=15)
        #plt.savefig(OUTPUT_PATH+"/"+fname+"rewards",bbox_inches='tight')
        #plt.close()


        plt.show()














if True:
    x0=[10]*12
    T=100
    ExpQuad.checkOffloading(x0,T)
    #ExpQuad.checkOffloadingVaryingAB(x0,T)
