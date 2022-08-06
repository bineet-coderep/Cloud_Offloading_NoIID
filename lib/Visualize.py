'''
Provides APIs to visualize various artifacts.
'''
import os,sys
PROJECT_ROOT = os.environ['CLD_OFLD_NO_IID_ROOT_DIR']
sys.path.append(PROJECT_ROOT)

from Parameters import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics as stat
import numpy as np
from scipy.optimize import curve_fit
import pylab

class Viz:

    def vizStats(trajs,labels,modelCosts,stateCosts,rewards,varyingABStats,th0=0,th1=1,fname="simple_system_"):
        plt.figure()
        for (traj,label,modelCost,stateCost) in zip(trajs,labels,modelCosts,stateCosts):
            X=[p[th0][0] for p in traj]
            Y=[p[th1][0] for p in traj]
            #plt.plot(X,Y,linestyle='--',markersize=4,marker='o',linewidth=1,label=label+" ("+str(modelCost)+", "+str("{0:.1f}".format(stateCost))+")")
            plt.plot(X,Y,linestyle='--',markersize=4,marker='o',linewidth=1,label=label)
        plt.legend()
        plt.savefig(OUTPUT_PATH+'/'+fname+"trajectories",bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.xlabel("Model Cost")
        plt.ylabel("Control Cost")
        plt.plot(varyingABStats[1],varyingABStats[2],color='#201230', marker='o', linestyle='dashed', alpha=0.4,linewidth=1, markersize=3,label="Varying Weights")
        for (label,modelCost,stateCost) in zip(labels,modelCosts,stateCosts):
            if label!='Random':
                plt.text(modelCost,stateCost,label,horizontalalignment='left',color='k')
                plt.plot(modelCost,stateCost,markersize=12,marker='o')
            else:
                meanModCost=stat.mean(modelCost)
                meanStateCost=stat.mean(stateCost)
                plt.scatter(modelCost,stateCost,s=70,marker='o',color='#a3dee3',alpha=0.3)
                plt.text(meanModCost,meanStateCost,label,horizontalalignment='left',color='k')
                plt.plot(meanModCost,meanStateCost,markersize=12,marker='o',color='#346e73')

        plt.legend()
        #plt.show()
        plt.savefig(OUTPUT_PATH+'/'+fname+"costs")
        plt.close()

        rewards_new=[]
        labels_new=[]
        for i in range(len(labels)):
            if labels[i]!='Random':
                rewards_new.append(rewards[i])
                labels_new.append(labels[i])
            else:
                for rwd in rewards[i]:
                    rewards_new.append(rwd)
                    labels_new.append("Random")

        rwdDict={
        'rewards':rewards_new,
        'policy':labels_new
        }
        rwds = pd.DataFrame(rwdDict)
        plt.figure()
        ax = sns.boxplot(x="policy", y="rewards", data=rwds)
        plt.ylabel('Rewards',fontweight="bold",fontsize=15)
        plt.xlabel('Policies',fontweight="bold",fontsize=15)
        plt.savefig(OUTPUT_PATH+"/"+fname+"rewards",bbox_inches='tight')
        plt.close()


        plt.show()


    def vizModelNonLinCost(choiceVarsList,losses,costsLin,costsQuad):
        plt.figure()
        plt.ylabel("Perception Accuracy (- Loss)")
        plt.xlabel("Perception Cost")

        acc=[-1*x for x in losses]

        plt.plot(costsQuad,acc,label="Quadratic Cost")
        plt.plot(costsLin,acc,label="Linear Cost")
        plt.legend()
        plt.show()

        plt.figure()
        plt.ylabel("Perception Accuracy (- Loss)")
        plt.xlabel("Choice Vars")

        acc=[-1*x for x in losses]

        plt.plot(choiceVarsList,acc,label="Quadratic Cost")

        plt.legend()
        plt.show()

        plt.figure()
        plt.ylabel("Cost")
        plt.xlabel("Choice Vars")

        acc=[-1*x for x in losses]

        plt.plot(choiceVarsList,costsLin,label="Linear Cost")
        plt.plot(choiceVarsList,costsQuad,label="Quadratic Cost")

        plt.legend()
        plt.show()


    def vizRealModels2(accs,costs,names):
        plt.figure()
        plt.ylabel("Accuracy")
        plt.xlabel("Cost")

        for (acc,cost,name) in zip(accs,costs,names):
            plt.scatter(cost,acc,label=name)
            model = np.poly1d(np.polyfit(cost, acc, (2)))
            polyline = np.linspace(1, max(cost), 50)
            plt.plot(polyline, model(polyline))
            model2 = np.poly1d(np.polyfit(cost, acc, (1/2)))
            polyline2 = np.linspace(1, max(cost), 50)
            plt.plot(polyline2, model2(polyline2))
            model3 = np.poly1d(np.polyfit(cost, acc, (1)))
            polyline3 = np.linspace(1, max(cost), 50)
            plt.plot(polyline3, model3(polyline3))
        plt.legend()
        plt.show()

    def vizRealModels(accs,costs,names,labels):
        plt.figure()
        plt.ylabel("Accuracy")
        plt.xlabel("Cost")

        for (cost,acc,label) in zip(costs,accs,labels):
            plt.scatter(cost,acc,label=label)

            '''model = np.poly1d(np.polyfit(cost, acc, (2)))
            polyline = np.linspace(1, max(cost), 50)
            plt.plot(polyline, model(polyline))'''

            popt, pcov = curve_fit(Viz.sigmoid, cost, acc, bounds=([-100, 100]), method='dogbox')

            '''
            efficientDetModels: popt, pcov = curve_fit(Viz.sigmoid, cost, acc, bounds=([-100, 100]), method='dogbox')
            efficientDetModels: popt, pcov = curve_fit(Viz.sigmoid, cost, acc, bounds=([-100, 100]), method='dogbox')
            '''

            print(*popt)

            #poptQuad, pcovQuad = curve_fit(Viz.quad, cost, acc, bounds=([-500, 500]), method='dogbox')

            x = np.linspace(2, 80, 50)
            y = Viz.sigmoid(x, *popt)

            #xQuad = np.linspace(5, 70, 50)
            #yQuad = Viz.quad(x, *poptQuad)

            plt.plot(x,y, label='Sig fit')
            #plt.plot(xQuad,yQuad, label='Quad fit')
            #plt.ylim(0, 1.05)

            for (a,c,name) in zip(acc,cost,names):
                #print(a,c,name)
                #plt.scatter([cost],[acc],s=100)
                plt.text(c,a,name,horizontalalignment='left',color='k')

        plt.legend()
        plt.show()

    def vizSimpleLossCost(accs,costs):
        plt.figure()
        plt.ylabel("Accuracy")
        plt.xlabel("Cost")


        plt.plot(costs,accs)
        plt.show()


    def sigmoidOld(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0)))+b
        return (y)

    def sigmoid(x, L ,x0, k, b):
        y = -97 / (1 + np.exp(-k*(x-x0)))+(b+93.3)
        return (y)

    def quad(x, a, b, c):
        y = (a*x*x)+(b*x)+c
        return (y)
