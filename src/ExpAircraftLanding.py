'''
Implements various functionalities using /lib/
'''
import os,sys,time
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *

from lib.Offloader import *
from lib.AircraftLanding import *
from lib.Visualize import *
from lib.Evaluater import *
#from lib.System import *
import random
import statistics as stat
from scipy.optimize import curve_fit



class ExpAircraftLanding:


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


    def checkOffloading(x0,T,TRIALS=100):
        toySystem=AircraftLanding.getSimpleSystem(x0,T)
        ofldr=OffloaderExpectation(toySystem)
        optChoices=ofldr.getOptOffloader()
        randChoices=ofldr.getRandOffloader()

        print(optChoices)
        #exit(0)

        ExpAircraftLanding.vizModelChoices(optChoices)

        evl=Evaluater(toySystem)



        #print(evl.getControlCost([1]*T))
        #print(evl.getControlCost([8]*T))








        # Evaluate
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

        oracleTraj=[]
        allCloudTraj=[]
        allRobotTraj=[]
        randTraj=[]
        optTraj=[]

        for i in range(TRIALS):
            evl=Evaluater(toySystem)
            #randChoices=ofldr.getRandOffloader()

            optRewards.append(evl.computeReward(optChoices))
            oracleRewards.append(evl.computeOracleReward())
            allCloudRewards.append(evl.computeReward([8]*T))
            allRobotRewards.append(evl.computeReward([1]*T))
            randRewards.append(evl.computeReward(randChoices))

            optPerCost.append(evl.getPerCost(optChoices))
            oraclePerCost.append(0)
            allCloudPerCost.append(evl.getPerCost([8]*T))
            allRobotPerCost.append(evl.getPerCost([1]*T))
            randPerCost.append(evl.getPerCost(randChoices))

            optCtrlCost.append(evl.getControlCost(optChoices))
            oracleCtrlCost.append(evl.getOracleControlCost())
            allCloudCtrlCost.append(evl.getControlCost([8]*T))
            allRobotCtrlCost.append(evl.getControlCost([1]*T))
            randCtrlCost.append(evl.getControlCost(randChoices))

            optTraj.append(evl.getTraj(optChoices))
            oracleTraj.append(evl.getOracleTraj())
            allCloudTraj.append(evl.getTraj([8]*T))
            allRobotTraj.append(evl.getTraj([1]*T))
            randTraj.append(evl.getTraj(randChoices))





        labels=["Oracle","Optimal","Random","All Large","All Small"]
        rewards=[oracleRewards,optRewards,randRewards,allCloudRewards,allRobotRewards]
        perCosts=[oraclePerCost,optPerCost,randPerCost,allCloudPerCost,allRobotPerCost]
        controlCosts=[oracleCtrlCost,optCtrlCost,randCtrlCost,allCloudCtrlCost,allRobotCtrlCost]
        trajs=[oracleTraj,optTraj,randTraj,allCloudTraj,allRobotTraj]


        print("> Stats")
        print("\t>> Labels: ", labels)
        print("\t>> Rewards: ", stat.mean(oracleRewards),stat.mean(optRewards),stat.mean(randRewards),stat.mean(allCloudRewards),stat.mean(allRobotRewards))
        print("\t>> Per Cost: ", stat.mean(oraclePerCost),stat.mean(optPerCost),stat.mean(randPerCost),stat.mean(allCloudPerCost),stat.mean(allRobotPerCost))
        print("\t>> Control Cost: ", stat.mean(oracleCtrlCost),stat.mean(optCtrlCost),stat.mean(randCtrlCost),stat.mean(allCloudCtrlCost),stat.mean(allRobotCtrlCost))

        ExpAircraftLanding.vizStats(labels,perCosts,controlCosts,rewards,trajs)


    def checkOffloadingVaryingAB(x0,T):
        toySystem=AircraftLanding.getSimpleSystem(x0,T)
        evl=Evaluater(toySystem)

        ctrlCosts=[]
        perCosts=[]


        #alpha=0.0
        alpha=1
        beta=0.01
        STEP=100

        for i in range(STEP):
            ofldr=OffloaderExpectation(toySystem)
            #optChoices=ofldr.getOptOffloader(alpha=alpha,beta=(1-alpha))
            optChoices=ofldr.getOptOffloader(alpha=alpha,beta=beta)
            randChoices=ofldr.getRandOffloader()

            perCosts.append(evl.getPerCost(optChoices))
            ctrlCosts.append(evl.getControlCost(optChoices))
            #alpha=alpha+0.01
            beta=beta*2

        print(perCosts)
        print(ctrlCosts)

        ExpAircraftLanding.vizParetoOpti(ctrlCosts,perCosts)

    def checkParetoAll(x0,T,TRIALS=100,STEPS=100):

        toySystem=AircraftLanding.getSimpleSystem(x0,T)

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

        optChoicesAB={}

        alpha_it=0
        for j in range(STEPS+1):
            ofldr=OffloaderExpectation(toySystem)
            optChoices=ofldr.getOptOffloader(alpha=alpha_it,beta=(1-alpha_it))
            optChoicesAB[(alpha_it,(1-alpha_it))]=optChoices
            alpha_it=alpha_it+(1/STEPS)

        #exit(0)


        for i in range(TRIALS+1):

            evl=Evaluater(toySystem)

            randChoices_it=ofldr.getRandOffloader()

            oraclePerCost.append(0)
            allCloudPerCost.append(evl.getPerCost([8]*T))
            allRobotPerCost.append(evl.getPerCost([1]*T))
            randPerCost.append(evl.getPerCost(randChoices_it))

            oracleCtrlCost.append(evl.getOracleControlCost())
            allCloudCtrlCost.append(evl.getControlCost([8]*T))
            allRobotCtrlCost.append(evl.getControlCost([1]*T))
            randCtrlCost.append(evl.getControlCost(randChoices_it))

            alpha_it=0

            ctrlCosts_it=[]
            perCosts_it=[]

            for j in range(STEPS+1):
                ofldr=OffloaderExpectation(toySystem)
                optChoices=optChoicesAB[(alpha_it,(1-alpha_it))]

                perCosts_it.append(evl.getPerCost(optChoices))
                ctrlCosts_it.append(evl.getControlCost(optChoices))
                alpha_it=alpha_it+(1/STEPS)

            optPerCost.append(perCosts_it)
            optCtrlCost.append(ctrlCosts_it)


        labels=["Oracle","Optimal","Random","All Large","All Small"]
        perCosts=[oraclePerCost,optPerCost,randPerCost,allCloudPerCost,allRobotPerCost]
        controlCosts=[oracleCtrlCost,optCtrlCost,randCtrlCost,allCloudCtrlCost,allRobotCtrlCost]


        #print(len(optCtrlCost[2]))
        ExpAircraftLanding.vizParetoOptiAll(labels,perCosts,controlCosts)




    def vizParetoOpti(ctrlCosts,perCosts):
        plt.figure()
        plt.ylabel("Control Cost",fontweight="bold",fontsize=21)
        plt.xlabel("Perception Cost",fontweight="bold",fontsize=21)


        plt.plot(perCosts,ctrlCosts,linewidth=2)
        plt.scatter(perCosts[0],ctrlCosts[0],s=100,color='black')
        plt.scatter(perCosts[-1],ctrlCosts[-1],s=100,color='green')
        plt.show()


    def vizParetoOptiAll(labels,perCosts,controlCosts,fname="aircraft_landing"):
        plt.figure()
        plt.ylabel("Control Cost",fontsize=21,fontweight="bold")
        plt.xlabel("Perception Cost",fontsize=21,fontweight="bold")

        clrs=plt.rcParams['axes.prop_cycle'].by_key()['color']

        for (label,perCost,controlCost,clr) in zip(labels,perCosts,controlCosts,clrs):
            if label=="Optimal":
                for (per,ctrl) in zip(perCost,controlCost):
                    plt.plot(per,ctrl,linewidth=3,color=clr,alpha=0.1)
                perMean=[]
                ctrlMean=[]
                perArray=np.array(perCost)
                ctrlArray=np.array(controlCost)
                ll=perArray.shape[1]
                for i in range(ll):
                    perMean.append(perArray[:,i])
                    ctrlMean.append(ctrlArray[:,i])

                plt.plot(per,ctrl,linewidth=5,label=label,color=clr)

        for (label,perCost,controlCost,clr) in zip(labels,perCosts,controlCosts,clrs):
            if label!="Optimal":
                meanPerCost=stat.mean(perCost)
                meanCtrlCost=stat.mean(controlCost)
                plt.scatter(perCost,controlCost,s=100,marker='o',alpha=0.1,color=clr)
                plt.plot(meanPerCost,meanCtrlCost,markersize=12,marker='o',label=label,color=clr)

        plt.legend()
        print("Printing!")
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+"pareto"+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)
        plt.close()





    def checkOffloadingContinous(x0,T):
        toySystem=AircraftLanding.getSimpleSystem(x0,T)
        ofldr=OffloaderExpectation(toySystem)
        optChoices=ofldr.getOptOffloaderContinous(cost=70)

        print(optChoices)

        ExpAircraftLanding.vizModelChoices(optChoices)


    def vizModelChoices(modelChoices,fname="model_selection_sequence"):
        modelChoices=[m-1 for m in modelChoices]
        plt.figure()
        plt.ylabel("Models",fontsize=21,fontweight="bold")
        plt.xlabel("Time",fontsize=21,fontweight="bold")
        X=list(range(len(modelChoices)))


        plt.plot(X,modelChoices,marker='o',markersize=10,linewidth=0.2,color='blue')
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+fname+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)

    def vizStats(labels,perCosts,controlCosts,rewards,trajs,fname="simple_system_"):
        plt.figure()

        plt.ylabel("Control Cost",fontweight="bold",fontsize=21)
        plt.xlabel("Perception Cost",fontweight="bold",fontsize=21)
        #plt.yticks(list(range(1080000,111000,100)))

        for (label,perCost,controlCost) in zip(labels,perCosts,controlCosts):
            #print(label)
            meanPerCost=stat.mean(perCost)
            meanCtrlCost=stat.mean(controlCost)
            plt.scatter(perCost,controlCost,s=100,marker='o',alpha=0.1)
            #plt.text(meanPerCost+(meanPerCost*0.04),meanCtrlCost+(meanCtrlCost*0.02),label,horizontalalignment='left',color='k',fontsize=15,fontweight="bold")
            plt.plot(meanPerCost,meanCtrlCost,markersize=15,marker='o')

        #plt.legend()
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+"cost_loss"+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)
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
        plt.ylabel('Rewards',fontweight="bold",fontsize=21)
        plt.xlabel('Policies',fontweight="bold",fontsize=21)
        #plt.savefig(OUTPUT_PATH+"/"+fname+"rewards",bbox_inches='tight')
        #plt.close()


        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+"rewards"+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)
        plt.close()

        plt.xlabel("Time",fontsize=20,fontweight="bold")
        plt.ylabel(r'$h$',fontsize=20,fontweight="bold")

        randTrial=random.randint(0,len(trajs[0])-1)

        for label,traj in zip(labels,trajs):
            #print(traj)
            '''if label=='Oracle' or label=='Cloud(7)':
                hList=[p[0][0] for p in traj[randTrial]]
                X=list(range(len(hList)))
                plt.plot(X,hList,linewidth=2,label=label)
            '''
            hListTmp=[p[0][0] for p in traj[randTrial]]
            hList=list(filter(lambda x: x>=-5, hListTmp))
            X=list(range(len(hList)))
            plt.plot(X,hList,linewidth=2,label=label)

        T=len(trajs[0][randTrial])
        plt.plot(list(range(T)),[0]*T,linewidth=2)

        plt.legend()
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+"traj_h"+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)
        plt.close()


        plt.xlabel("Time",fontsize=21,fontweight="bold")
        plt.ylabel(r'$\theta$',fontsize=21,fontweight="bold")

        randTrial=random.randint(0,len(trajs[0])-1)

        for label,traj in zip(labels,trajs):
            #print(traj)
            '''if label=='Oracle' or label=='Cloud(7)':
                thetaList=[p[2][0] for p in traj[randTrial]]
                X=list(range(len(thetaList)))
                plt.plot(X,thetaList,linewidth=2,label=label)
            '''
            thetaList=[p[2][0] for p in traj[randTrial]]
            X=list(range(len(thetaList)))
            plt.plot(X,thetaList,linewidth=2,label=label)

        T=len(trajs[0][randTrial])
        plt.plot(list(range(T)),[0]*T,linewidth=2)

        plt.legend()
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+"traj_theta"+'.pdf', format='pdf',bbox_inches='tight',pad_inches = 0,transparent = True)
        plt.close()

















if True:
    #x0=[1000,2,20,2]
    #x0=[(900,1100),(1,10),(1,30),(1,3)]
    #x0=[-100,-2,-20,-2]
    #x0=[0.001,0.00002,0.001,0.00002]
    #x0=[(100,100),(1,1),(10,10),(0.5,0.5)]
    x0=[(1000,1000),(2,2),(2,2),(0.2,0.2)]
    T=150
    TRIALS=100
    STEPS=90
    ExpAircraftLanding.checkOffloading(x0,T,TRIALS=TRIALS)
    #ExpAircraftLanding.checkOffloadingVaryingAB(x0,T)
    #ExpAircraftLanding.checkParetoAll(x0,T,TRIALS=TRIALS,STEPS=STEPS)
