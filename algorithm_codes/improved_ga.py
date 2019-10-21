
#%%
#Initial population: Unrestricted (totally random)
#Mutation: Switch one gene only

####This version Unrestricted + Switch Only ######

#For all variants we use following GA parameters:
#Population: 100
#Generation: 50
#Crossover:0.8
#Mutation:0.1 (just select one gene to mutate)


#%%
import numpy as np
import math
import sys
import json
#import sframe as sf
#from bokeh.charts import BoxPlot, output_notebook, show
#import seaborn as sns
#import pulp
#from pulp import *
import time
#from gurobipy import *
import random

from deap import base
from deap import creator
from deap import tools


#%%
#Below functions are needed for Queuing approximation

# Define the generator that will generate all possible assignments of classes to servers, without permutations
def generateVectorsFixedSum(m,n):
    # generator for all combinations of $w$ for given number of servers and classes
    if m==1:
        yield [n]
    else:
        for i in range(n+1):
            for vect in generateVectorsFixedSum(m-1,n-i):
                yield [i]+vect

def MMCsolver(lamda, mu, nservers, mClasses):
    assert sum(lamda/mu)<nservers # ensure stability
    
    # initialize \Lamda and \alpha
    lambdaTot = sum(lamda)
    alpha = lamda/lambdaTot
    
    # create mapping between the combination vectors and matrix columns/rows
    idx_map = dict([ (tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, nservers)) ])
    # need to use tuple here as 'list' cannot be as a key
    i_map   = dict([(idx_map[idx], list(idx)) for idx in idx_map ]) 
    # need to use list here as 'tuple' cannot be modified as will be need further
    
    # function to get matrix index based on the system state
    def getIndexDict(idx, idx_map):
        try:
            return idx_map[tuple(idx)]
        except KeyError:
            return -1
    # generate matrices A_0 and A_1
    q_max = len(idx_map)
    A0 = np.zeros((q_max,q_max))  #corresponds to terms with i items in queue
    A1 = np.zeros((q_max,q_max))  #corresponds to terms with i+1 items in queue
    for i, idx in i_map.items():
        #diagonal term
        A0[i,i] += 1 + np.sum(idx*mu)/lambdaTot

    #term corresponding to end of service for item j1, start of service for j2
        for j1 in range(mClasses):
            for j2 in range(mClasses):
                idx[j1] += 1; idx[j2] -= 1
                i1 = getIndexDict(idx, idx_map)  #convert 'list' back to tuple to use it as a key
                if i1>=0: A1[i,i1] += alpha[j2]/lambdaTot*idx[j1]*mu[j1]
                idx[j1] -= 1; idx[j2] += 1
    
    # compute matrix Z iteratively
    eps = 0.00000001
    I = np.eye(q_max) # produces identity matrix
    Z_prev = np.zeros((q_max, q_max)) 
    delta=1
    A0_inv = np.linalg.inv(A0)
    while delta>eps:
        Z = np.dot(A0_inv, I + np.dot(A1, np.dot(Z_prev, Z_prev)))  #invA0*(I+A1*Z*Z)
        delta = np.sum(np.abs(Z-Z_prev))
        Z_prev=Z
    
    # generate Q matrices, it will be stored in a list
    Q = []
    idxMat = [] # matrix with server occupancy for each system state, will be used in computing the system parameters
    Q.insert(0, Z[:])
    idxMat.insert(0, np.array([x for x in i_map.values()]))
    
    i_map_full = []
    i_map_full.append(i_map)


    idx_map_nplus = idx_map #dict([ (tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, nServers)) ])
    i_map_nplus   = i_map #dict([(idx_map_nplus[idx], list(idx)) for idx in idx_map_nplus ])
    q_max_nplus   = len(idx_map_nplus)

    idx_map_n = idx_map_nplus
    i_map_n   = i_map_nplus
    q_max_n   = q_max_nplus

    A1_n = A1[:]

    for n in range(nservers,0,-1):
        idx_map_nminus = dict([ (tuple(vect), i) for i, vect in enumerate(generateVectorsFixedSum(mClasses, n-1)) ])
        i_map_nminus   = dict([(idx_map_nminus[idx], list(idx)) for idx in idx_map_nminus ])
        q_max_nminus   = len(idx_map_nminus)
    
        i_map_full.insert(0,i_map_nminus)


        L_n = np.zeros((q_max_n,q_max_nminus))  #corresponds to terms with i items in queue
        A0_n = np.zeros((q_max_n,q_max_n))  #corresponds to terms with i items in queue
        for i, idx in i_map_n.items():

            #diagonal term
            A0_n[i,i] += 1 + np.sum(idx*mu)/lambdaTot

            #term corresponding to arrival of item item j1
            for j2 in range(mClasses):
                idx[j2] -= 1
                i2 = getIndexDict(idx, idx_map_nminus) 
                if i2>=0: L_n[i,i2] += alpha[j2]
                idx[j2] += 1

        # Q_n = (A_0 - A_1*Q_{n+1})^{-1}*L_n
        Q.insert(0, np.dot(np.linalg.inv(A0_n-np.dot(A1_n, Q[0])), L_n)) 

        idx_map_nplus = idx_map_n
        i_map_nplus   = i_map_n
        q_max_nplus   = q_max_n

        idx_map_n = idx_map_nminus
        i_map_n   = i_map_nminus
        q_max_n   = q_max_nminus
        idxMat.insert(0, np.array([x for x in i_map_n.values()]))    


        A1_n = np.zeros((q_max_n,q_max_nplus))  #corresponds to terms with i+1 items in queue
        for i, idx in i_map_n.items():
            #term corresponding to end of service for item j1
            for j1 in range(mClasses):
                idx[j1] += 1
                i1 = getIndexDict(idx, idx_map_nplus) 
                if i1>=0: A1_n[i,i1] += idx[j1]*mu[j1]/lambdaTot
                idx[j1] -= 1
    
    # compute the P_n for n<k and normalize it such that sum(P_n) = 1
    P = []
    P.append([1.0])

    sm = 1.0
    for n in range(nservers):
        P.append(np.dot(Q[n],P[-1]))
        sm += sum(P[-1])

    sm += sum(np.dot(np.linalg.inv(np.eye(len(P[-1])) - Z), np.dot(Z,P[-1])))

    for p in P: p[:] /= sm  #normalization
        
        
    # compute totals needed for the E[Q_i] - marginal distributions
    inv1minZ = np.linalg.inv(np.eye(len(P[-1])) - Z)
    EQTotal = sum(np.dot(np.dot(np.dot(inv1minZ,inv1minZ), Z),P[-1]))
    EQQmin1Total = 2*sum(np.dot(np.dot(np.dot(np.dot(np.dot(inv1minZ,inv1minZ),inv1minZ), Z), Z), P[-1]))
    EQ2Total = EQQmin1Total + EQTotal

    # compute 1st and 2nd marginal moments of the numbers in the queue E[Q_i] and E[Q_i^2]
    EQ = alpha*EQTotal
    EQQmin1 = alpha*alpha*EQQmin1Total
    EQ2 = EQQmin1 + EQ
    
    # compute 1st and 2nd marginal moments of the numbers in the system E[N_i] and E[N_i^2]
    ENTotal = EQTotal + sum(lamda/mu)
    EN = EQ + lamda/mu

    # TODO compute the E[N_i^2]
    ES2 = np.zeros(mClasses)
    for (p, idx) in zip(P[:-1], idxMat[:-1]):
        ES2 += np.dot(p, idx**2)
    ES2 += np.dot(np.dot(inv1minZ, P[-1]), idxMat[-1]**2)

    ESq = alpha*np.dot(np.dot(np.dot(np.dot(inv1minZ,inv1minZ), Z),P[-1]), idxMat[-1])

    EN2 = EQ2 + 2*ESq + ES2
    
    # compute marginal variances of the numbers in the queue Var[Q_i] and in the system Var[N_i] 
    VarQTotal = EQ2Total - EQTotal**2
    VarQ = EQ2 - EQ**2

    VarN = EN2 - EN**2
    
    # computeMarginalDistributions
    qmax=1500

    marginalN = np.zeros((mClasses, qmax))

    for m in range(mClasses):
        for imap, p in zip(i_map_full[:-1], P[:-1]):
            for i, idx in imap.items():
                marginalN[m,idx[m]] += p[i]

        inv1minAlphaZ = np.linalg.inv(np.eye(len(P[-1])) - (1-alpha[m])*Z)
        frac = np.dot(alpha[m]*Z, inv1minAlphaZ)
        # tmp = np.dot(self.Z, self.P[-1])
        # tmp = np.dot(inv1minAlphaZ, tmp)
        tmp = np.dot(inv1minAlphaZ, P[-1])

        for q in range(0,qmax):
            for i, idx in i_map_full[-1].items():
                if idx[m]+q < qmax: marginalN[m,idx[m]+q] += tmp[i]
            tmp = np.dot(frac, tmp)
    return marginalN, EN, VarN


def whittApprox(E1, E2, E3):
    '''
    input: first 3 moments of hyperexpo dist.
    returns: parameters of hyperexpo (p, v1 and v2)
    uses whitt approximation.....
    '''
    x=E1*E3-1.5*E2**2
    #print x
    assert x>=0.0
    
    y=E2-2*(E1**2)
    #print y
    assert y>=0.0
    
    Ev1= ((x+1.5*y**2+3*E1**2*y)+math.sqrt((x+1.5*y**2-3*E1**2*y)**2+18*(E1**2)*(y**3)))/(6*E1*y)
    #print Ev1
    assert Ev1>=0
    
    Ev2= ((x+1.5*y**2+3*E1**2*y)-math.sqrt((x+1.5*y**2-3*E1**2*y)**2+18*(E1**2)*(y**3)))/(6*E1*y)
    assert Ev2>=0
    
    
    p=(E1-Ev2)/(Ev1-Ev2)
    assert p>=0
    
    return 1.0/Ev1, 1.0/Ev2, p

def isServiceRateEqual(mu):
    return len(set(mu)) <= 1

def Approx_MMCsolver(lamda, mu, nServers, mClasses):
    '''
    inputs: lamda->failure rates of SKUs
            mu ->service rates of servers for SKUs
            nServers->number of servers in repairshop
            mClasses->number of SKUs := length of failure rates
            
    output: Marginal Queue length for each type of SKU
            Expected Queue length  ''  ''   ''   ''
            Variance of Queue length ''  '' ''   ''
    
    solution: Approximate 3 class system and calls MMCsolver
    '''
    
    marginalN=[]
    EN=[]
    VarN=[]
    
    for mCl in range(mClasses):
        #first moment for service time distribution for approximation:
        E_S1=(np.inner(lamda, 1/mu)-(lamda[mCl]*1/mu[mCl]))/(sum(lamda)-lamda[mCl]) #checked
       
        #second moment
        E_S2=2*(np.inner(lamda, (1/mu)**2)-(lamda[mCl]*(1/mu[mCl])**2))/(sum(lamda)-lamda[mCl]) #checked
    
        #third moment 
        E_S3=6*(np.inner(lamda, (1/mu)**3)-(lamda[mCl]*(1/mu[mCl])**3))/(sum(lamda)-lamda[mCl]) #checked
   
    
        #calculate inputs for to check neccesity condtion:
        varA=E_S2-E_S1**2
        cA=math.sqrt(varA)/E_S1
    
        #to check if all of the service rates of approximated service are same
        #if it is true sum of hyperexpo with same parameter is ---> exponential distribution
    
        mu_copy=[]
        mu_copy[:]=mu
        del mu_copy[mCl]
    
        if isServiceRateEqual(mu_copy)==True:
            #we can assume there is only aggreate remaing streams to one rather than two
            p=1
            v1=mu_copy[0]
        
            lam1=lamda[mCl]
            #S1=1/mu[mCl]
    
            lamA1=p*(sum(lamda)-lamda[mCl])
            #SA1=1/float(v1)
            
            #if sum(lamda/mu)>nservers:
            #    nservers
            #we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength= MMCsolver(np.array([lam1, lamA1]),                                  np.array([mu[mCl], v1]), nservers=nServers, mClasses=2)
    
            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])
    
        #if (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3)<0.0:
        #E_S3=(3.0/2.0)*((1+cA**2)**2)*E_S1**3+0.01
        #    print "aaa"
        #    v1, v2, p=whittApprox(E_S1, E_S2, E_S3)
    
        else:
            #a2 calculation
            a2=(6*E_S1-(3*E_S2/E_S1))/((6*E_S2**2/4*E_S1)-E_S3)
        
            #a1 calculation
            a1=(1/E_S1)+(a2*E_S2/(2*E_S1))
    
            #v1 calculation
            v1=(1.0/2.0)*(a1+math.sqrt(a1**2-4*a2))
        
            #v2 calculation
            v2=(1.0/2.0)*(a1-math.sqrt(a1**2-4*a2))
    
            #p calculation
            p=1-((v2*(E_S1*v1-1))/float((v1-v2)))
    
            lam1=lamda[mCl]
            #S1=1/mu[mCl]
    
            lamA1=p*(sum(lamda)-lamda[mCl])
            #SA1=1/float(v1)
   
    
            lamA2=(1-p)*(sum(lamda)-lamda[mCl])
            #SA2=1/float(v2)
    
    
            #Now we have 3 classes of streams (2 streams for approximation) as usual
            #so mClasses=3
        
            marginalLength, ENLength, VarLength = MMCsolver(np.array([lam1, lamA1, lamA2]),                                  np.array([mu[mCl], v1, v2]), nservers=nServers, mClasses=3)
    
            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])
   
    return marginalN, EN, VarN

def Approx_MMCsolver2(lamda, mu, nservers, mClasses):
    '''
    inputs: lamda->failure rates of SKUs
            mu ->service rates of servers for SKUs
            nservers->number of servers in repairshop
            mClasses->number of SKUs := length of failure rates
            
    output: Marginal Queue length for each type of SKU
            Expected Queue length  ''  ''   ''   ''
            Variance of Queue length ''  '' ''   ''
    
    solution: Approximate 3 class system and calls MMCsolver
    '''
    
    #print nservers
    marginalN=[]
    EN=[]
    VarN=[]
    
    for mCl in range(mClasses):
        #first moment for service time distribution for approximation:
        E_S1=(np.inner(lamda, 1/mu)-(lamda[mCl]*1/mu[mCl]))/(sum(lamda)-lamda[mCl]) #checked
        #print E_S1
        #second moment
        E_S2=2*(np.inner(lamda, (1/mu)**2)-(lamda[mCl]*(1/mu[mCl])**2))/(sum(lamda)-lamda[mCl]) #checked
    
        #third moment 
        E_S3=6*(np.inner(lamda, (1/mu)**3)-(lamda[mCl]*(1/mu[mCl])**3))/(sum(lamda)-lamda[mCl]) #checked
   
    
        #calculate inputs for to check neccesity condtion:
        varA=E_S2-E_S1**2
        cA=math.sqrt(varA)/E_S1
        
        
        assert (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3) > 0
            
    
        #to check if all of the service rates of approximated service are same
        #if it is true sum of hyperexpo with same parameter is ---> exponential distribution
    
        mu_copy=[]
        mu_copy[:]=mu
        del mu_copy[mCl]
    
        if isServiceRateEqual(mu_copy)==True:
            #we can assume there is only aggreate remaing streams to one rather than two
            p=1
            v1=mu_copy[0]
        
            lam1=lamda[mCl]
            #S1=1/mu[mCl]
    
            lamA1=p*(sum(lamda)-lamda[mCl])
            #SA1=1/float(v1)
            
            #sum(lamda/mu)<nservers
            
            if sum(np.array([lam1, lamA1])/np.array([mu[mCl], v1]))>nservers:
                #print "hasan"
                nservers=int(sum(np.array([lam1, lamA1])/np.array([mu[mCl], v1])))+1
        
            #we have only two streams now so mClasses=2
            marginalLength, ENLength, VarLength= MMCsolver(np.array([lam1, lamA1]),                                  np.array([mu[mCl], v1]), nservers, mClasses=2)
    
            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])
            #print "aaaa"
    
        #if (E_S3-(3.0/2.0)*((1+cA**2)**2)*E_S1**3)<0.0:
        #E_S3=(3.0/2.0)*((1+cA**2)**2)*E_S1**3+0.01
        #    print "aaa"
        #    v1, v2, p=whittApprox(E_S1, E_S2, E_S3)
    
        else:
            
            v1, v2, p=whittApprox(E_S1, E_S2, E_S3)
            #print v1
            #print v2
            
            lam1=lamda[mCl]
            #S1=1/mu[mCl]
    
            lamA1=p*(sum(lamda)-lamda[mCl])
            #SA1=1/float(v1)
   
    
            lamA2=(1-p)*(sum(lamda)-lamda[mCl])
            #SA2=1/float(v2)
    
            if sum(np.array([lam1, lamA1, lamA2])/np.array([mu[mCl], v1, v2]))>=nservers:
                #print "turan"
                nservers=int(sum(np.array([lam1, lamA1, lamA2])/np.array([mu[mCl], v1, v2])))+1
            #Now we have 3 classes of streams (2 streams for approximation) as usual
            #so mClasses=3
        
            marginalLength, ENLength, VarLength = MMCsolver(np.array([lam1, lamA1, lamA2]),                                  np.array([mu[mCl], v1, v2]), nservers, mClasses=3)
    
            marginalN.append(marginalLength[0])
            EN.append(ENLength[0])
            VarN.append(VarLength[0])
   
    return marginalN, EN, VarN, nservers


#%%
#code for optimization inventories after given queue length distribution
def OptimizeStockLevelsAndCosts(holdingCosts, penalty, marginalDistribution):

    if not isinstance(holdingCosts, np.ndarray):
        holdingCosts = np.array(holdingCosts)

    if len(marginalDistribution.shape) == 1:
        marginalDistribution = marginalDistribution.reshape(1,len(marginalDistribution))

    nSKUs=len(holdingCosts)
    maxQueue = marginalDistribution.shape[1]
    n_array = np.array(range(maxQueue))
    S = np.zeros(nSKUs, dtype=int)
    PBO = np.sum(marginalDistribution[:,1:], axis=1)
    EBO = np.sum(marginalDistribution*np.array(range(marginalDistribution.shape[1])), axis=1)

    hb_ratio = holdingCosts/penalty
    for sk in range(nSKUs):
        while S[sk]<maxQueue and np.sum(marginalDistribution[sk, S[sk]+1:]) > hb_ratio[sk]:
            S[sk] += 1
            PBO[sk] = np.sum(marginalDistribution[sk,S[sk]+1:]) # -= marginalDistribution[sk, S[sk]]
            EBO[sk] = np.sum(marginalDistribution[sk,S[sk]:]*n_array[:-S[sk]]) #-= PBO[sk]

    totalCost = np.sum(S*holdingCosts) + np.sum(penalty*EBO)
    hCost=np.sum(S*holdingCosts)
    pCost=np.sum(penalty*EBO)
    #print ((EBO < 0).sum() == EBO.size).astype(np.int)
    #if pCost<0.0:
    #    print EBO
        #print  ((EBO < 0).sum() == EBO.size).astype(np.int)
        #print all(i >= 0.0 for i in marginalDistribution)
    
    return totalCost, hCost, pCost, S, EBO


#%%
def individual2cluster(individual):
    '''
    -input: list of integers representing assingment of SKUs to clusters
    -output: list of list representing clusters and assinged SKUs in each cluster
    '''
    a=[]
    for i in range(len(individual)):
        a.append((i+1, individual[i]))
    values = set(map(lambda x:x[1], a))
    newlist = [[y[0] for y in a if y[1]==x] for x in values]
    return newlist


#%%
def evalOneMax(FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost, individual):
    '''
    input: -Individual representing clustering scheme
           -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -MMCsolver and Approx_MMCsolver functions--> to find Queue length dist. of failed SKUs
                                                    --> number of SKUS >=4 use approximation
           
           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)
           
     output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU, # of 
             servers at each cluster
    
     evalOneMax function evaluates the fitness of individual chromosome by:
           (1) chromosome converted a clustering scheme
           (2) for each SKU in each cluster at the clustering scheme queue length dist. evaluated by calling MMC solver
           (3) OptimzeStockLevels function is called by given queue length dist. and initial costs are calculated
           (4) Local search is performed by increasing server numbers in each cluster by one and step (2) and (3) repeted
           (5) Step (4) is repated if there is a decrease in total cost
     
    
    Warning !! type matching array vs list might be problem (be careful about type matching)
    
    '''
    
    #from individual to cluster 
    cluster_GA=individual2cluster(individual)
    #bestCost=float('inf')
    #bestCluster=[]
    bestS=[]
    bestEBO=[]
    EBO_cluster=[]
    S_cluster=[]
    bestserverAssignment=[]
    serverAssignment=[]
    sliceIndex2=[]
    TotalCost=0.0
    TotalHolding, TotalPenalty, TotalSkillCost, TotalMachineCost=0.0, 0.0, 0.0, 0.0
    #LogFileList=[]
    #logFile={} 
    #iterationNum=0
    for cluster in cluster_GA:
        sliceIndex2[:]=cluster
        sliceIndex2[:] = [x - 1 for x in sliceIndex2]
        
        sRate=np.array(ServiceRates[sliceIndex2])
        fRate=np.array(FailureRates[sliceIndex2])
        hcost=np.array(holding_costs[sliceIndex2])
        
        min_nserver=int(sum(fRate/sRate))+1
            
            #costTemp=0
            #while costTemp<=machineCost:
        if len(sRate)<=3:
            marginalDist, EN, VarN=MMCsolver(fRate, sRate, min_nserver, len(fRate))
        else:
            marginalDist, EN, VarN, min_nserverUpdate=Approx_MMCsolver2(fRate, sRate, min_nserver, len(fRate))
            min_nserver=min_nserverUpdate
        
        totalCostClust, hCost, pCost, S, EBO=OptimizeStockLevelsAndCosts(hcost, penalty_cost, np.array(marginalDist))
            
            #increasing number of servers and checking if total cost decreases
            
        TotalMachine_Cost=min_nserver*machineCost
        TotalSkill_Cost=min_nserver*len(fRate)*skillCost
            
        totalCostClust=totalCostClust+TotalMachine_Cost+TotalSkill_Cost
            
        while  True:
            min_nserver+=1
            if len(sRate)<=3:
                marginalDist, EN, VarN=MMCsolver(fRate, sRate, min_nserver, len(fRate))
            else:
                marginalDist, EN, VarN, min_nserverUpdate=Approx_MMCsolver2(fRate, sRate, min_nserver, len(fRate))
                min_nserver=min_nserverUpdate
        
            temp_totalCostClust, temp_hCost, temp_pCost, temp_S, temp_EBO=OptimizeStockLevelsAndCosts(hcost, penalty_cost, np.array(marginalDist))
            temp_TotalMachine_Cost=min_nserver*machineCost
            temp_TotalSkill_Cost=min_nserver*len(fRate)*skillCost
                
            temp_totalCostClust=temp_totalCostClust+temp_TotalMachine_Cost+temp_TotalSkill_Cost
                
            if temp_totalCostClust> totalCostClust:
                min_nserver-=1
                break
            else:
                totalCostClust=temp_totalCostClust
                    
                TotalMachine_Cost=temp_TotalMachine_Cost
                TotalSkill_Cost=temp_TotalSkill_Cost
                hCost=temp_hCost
                pCost=temp_pCost
                
            
            
        TotalHolding+=hCost
        TotalPenalty+=pCost
       
        TotalSkillCost+=TotalSkill_Cost
        TotalMachineCost+=TotalMachine_Cost
        
        TotalCost=TotalCost+totalCostClust
        
        EBO_cluster.append(EBO.tolist())
        S_cluster.append(S.tolist())
        serverAssignment.append(min_nserver)

    return TotalCost, 

#bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster, bestS, bestEBO, \
#            bestserverAssignment, LogFileList
#DONT FORGET COME AT THE END!!!
        


#%%
def Final_evalOneMax(FailureRates, ServiceRates, holding_costs, penalty_cost, skillCost, machineCost, individual):
    '''
    input: -Individual representing clustering scheme
           -Failure rates and corresponding service rates of each SKU
           -Related cost terms holding costs for SKUs(array), backorder, skill and server (per server and per skill)
           -MMCsolver and Approx_MMCsolver functions--> to find Queue length dist. of failed SKUs
                                                    --> number of SKUS >=4 use approximation
           
           -OptimizeStockLevels calculates EBO and S for giving clustering (Queue length dist.)
           
     output: Returns best total cost and other cost terms, Expected backorder (EBO) and stocks (S) for each SKU, # of 
             servers at each cluster
    
     evalOneMax function evaluates the fitness of individual chromosome by:
           (1) chromosome converted a clustering scheme
           (2) for each SKU in each cluster at the clustering scheme queue length dist. evaluated by calling MMC solver
           (3) OptimzeStockLevels function is called by given queue length dist. and initial costs are calculated
           (4) Local search is performed by increasing server numbers in each cluster by one and step (2) and (3) repeted
           (5) Step (4) is repated if there is a decrease in total cost
     
    
    Warning !! type matching array vs list might be problem (be careful about type matching)
    
    '''
    
    #from individual to cluster 
    cluster_GA=individual2cluster(individual)
    #bestCost=float('inf')
    #bestCluster=[]
    bestS=[]
    bestEBO=[]
    EBO_cluster=[]
    S_cluster=[]
    bestserverAssignment=[]
    serverAssignment=[]
    sliceIndex2=[]
    TotalCost=0.0
    TotalHolding, TotalPenalty, TotalSkillCost, TotalMachineCost=0.0, 0.0, 0.0, 0.0
    #LogFileList=[]
    #logFile={} 
    #iterationNum=0
    for cluster in cluster_GA:
        sliceIndex2[:]=cluster
        sliceIndex2[:] = [x - 1 for x in sliceIndex2]
        
        sRate=np.array(ServiceRates[sliceIndex2])
        fRate=np.array(FailureRates[sliceIndex2])
        hcost=np.array(holding_costs[sliceIndex2])
        
        min_nserver=int(sum(fRate/sRate))+1
            
            #costTemp=0
            #while costTemp<=machineCost:
        if len(sRate)<=3:
            marginalDist, EN, VarN=MMCsolver(fRate, sRate, min_nserver, len(fRate))
        else:
            marginalDist, EN, VarN, min_nserverUpdate=Approx_MMCsolver2(fRate, sRate, min_nserver, len(fRate))
            min_nserver=min_nserverUpdate
        
        totalCostClust, hCost, pCost, S, EBO=OptimizeStockLevelsAndCosts(hcost, penalty_cost, np.array(marginalDist))
            
            #increasing number of servers and checking if total cost decreases
            
        TotalMachine_Cost=min_nserver*machineCost
        TotalSkill_Cost=min_nserver*len(fRate)*skillCost
            
        totalCostClust=totalCostClust+TotalMachine_Cost+TotalSkill_Cost
            
        while  True:
            min_nserver+=1
            if len(sRate)<=3:
                marginalDist, EN, VarN=MMCsolver(fRate, sRate, min_nserver, len(fRate))
            else:
                marginalDist, EN, VarN, min_nserverUpdate=Approx_MMCsolver2(fRate, sRate, min_nserver, len(fRate))
                min_nserver=min_nserverUpdate
        
            temp_totalCostClust, temp_hCost, temp_pCost, temp_S, temp_EBO=OptimizeStockLevelsAndCosts(hcost, penalty_cost, np.array(marginalDist))
            temp_TotalMachine_Cost=min_nserver*machineCost
            temp_TotalSkill_Cost=min_nserver*len(fRate)*skillCost
                
            temp_totalCostClust=temp_totalCostClust+temp_TotalMachine_Cost+temp_TotalSkill_Cost
                
            if temp_totalCostClust> totalCostClust:
                min_nserver-=1
                break
            else:
                totalCostClust=temp_totalCostClust
                    
                TotalMachine_Cost=temp_TotalMachine_Cost
                TotalSkill_Cost=temp_TotalSkill_Cost
                hCost=temp_hCost
                pCost=temp_pCost
                
            
            
        TotalHolding+=hCost
        TotalPenalty+=pCost
       
        TotalSkillCost+=TotalSkill_Cost
        TotalMachineCost+=TotalMachine_Cost
        
        TotalCost=TotalCost+totalCostClust
        
        EBO_cluster.append(EBO.tolist())
        S_cluster.append(S.tolist())
        serverAssignment.append(min_nserver)

    return TotalCost, TotalHolding, TotalPenalty, TotalMachineCost, TotalSkillCost, cluster_GA , S_cluster, EBO_cluster,           serverAssignment
#DONT FORGET COME AT THE END!!!
        


#%%
#change mutation 
def swicthtoOtherMutation(individual):
    
    individual_new = individual[:]

    numSKUs = len(individual_new)
    
    idx1= random.sample(range(0, numSKUs), 1)[0]
    
    #print idx1
    
    #print individual[idx1]
    
    
    ex_cluster_number = individual[idx1]
    
    numbers = [x for x in range(1,numSKUs+1) if x != ex_cluster_number]
    
    individual_new[idx1] = random.choice(numbers)
    
    return creator.Individual(individual_new)


#%%
#    This file is part of DEAP.
#
#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1


def GAPoolingHeuristic(failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost,numSKUs,minCluster,maxCluster): 

    # 1 is for maximization -1 for minimization
    # Minimize total cost
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def generateIndividual(numSKUs, minCluster,maxCluster):
    #Generating initial indvidual that are in the range of given max-min cluster numbers
    
        return creator.Individual(np.random.choice(np.arange(minCluster, maxCluster+1), numSKUs))
        
    
    toolbox = base.Toolbox()

    # Attribute generator 
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [1,number of SKUs] (i.e. 0 or 1 with equal
    #                      probability)
    
    #toolbox.register("attr_bool", random.randint, 1, len(failure_rates))

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of #number of maximum cluster =#of SKUs 'attr_bool' elements ('genes')
    toolbox.register("individual", generateIndividual, numSKUs,minCluster,maxCluster)

    
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized
    #for objective function call pooling optimizer !!!
    # what values need for optimizer !!!
    
    
    #def evalOneMax(individual):
    #    return sum(individual),

    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalOneMax, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost)

    # register the crossover operator
    toolbox.register("mate", tools.cxOnePoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    #
    toolbox.register("mutate", swicthtoOtherMutation)
    #toolbox.register("mutate", swicthtoOtherMutation)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=10)

    #----------

#def main():
    random.seed(64)

    # create an initial population of 50 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=100)
    
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.7, 0.10
    
    #ADD Hall of Fame for the next version !!!
    #hof = tools.HallOfFame(1)
    #print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    #print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    #while max(fits) < 100 and g < 100:
    while g < 50:
        # A new generation
        g = g + 1
        #print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                mutant=toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        #print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        #length = len(pop)
        #mean = sum(fits) / length
        #sum2 = sum(x*x for x in fits)
        #std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        
    #    print("  Max %s" % max(fits))
    #    print("  Avg %s" % mean)
    #    print("  Std %s" % std)
    
    #print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (individual2cluster(best_ind), best_ind.fitness.values))
    
    return best_ind.fitness.values, best_ind

#if __name__ == "__main__":
#    main()


#%%
json_case=[]
with open("fullRangeResultsFullFlexNew.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case.append(json.loads(line))

#for case in json_case:
#    print case["caseID"]


#%%
results=[]
GAPoolingResult={}

for case in json_case:
    if case["caseID"]!="Case: 000x":
        failure_rates=case["failure_rates"]
        service_rates=case["service_rates"]
        holding_costs=case["holding_costs"]
        skill_cost=case["skill_cost"]
        penalty_cost=case["penalty_cost"]
        machine_cost=case["machine_cost"]
    
        start_time = time.time()
        
        #unrestricted initial population _v4a
        numSKUs,minCluster,maxCluster=len(failure_rates), 1, len(failure_rates)

        _, best_ind=GAPoolingHeuristic(np.array(failure_rates), np.array(service_rates),                                                   np.array(holding_costs), penalty_cost, skill_cost, machine_cost,numSKUs,minCluster,maxCluster)
        stop_time=time.time()-start_time
        
        
        
        #best individual is ran one more the for statistical data collection and recording
        bestCost, bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster , bestS, bestEBO,           bestserverAssignment=Final_evalOneMax(np.array(failure_rates), np.array(service_rates), np.array(holding_costs),                           penalty_cost, skill_cost, machine_cost, best_ind)
        
        
        
        
        
        
        print (case["caseID"], bestCost)
        GAPoolingResult["caseID"]=case["caseID"]
    
        GAPoolingResult["GAPoolingruntime"]=stop_time
        GAPoolingResult["GAPoolingTotalCost"]=bestCost
        GAPoolingResult["GAPoolingHoldingCost"]=bestHolding
        GAPoolingResult["GAPoolingPenaltyCost"]=bestPenalty
        GAPoolingResult["GAPoolingMachineCost"]=bestMachineCost
        GAPoolingResult["GAPoolingSkillCost"]=bestSkillCost
            
        GAPoolingResult["GAPoolingCluster"]=bestCluster
        GAPoolingResult["GAPoolingS"]=bestS
        GAPoolingResult["GAPoolingEBO"]=bestEBO
        GAPoolingResult["GAPoolingServerAssignment"]=bestserverAssignment
        #KmedianResult["KmedianLogFile"]=LogFileList
            
        GAPoolingResult["GAP"]=bestCost-case["total_cost"]
        GAPoolingResult["GAPoolingPercentGAP"]=100*(bestCost-case["total_cost"])/case["total_cost"]
            
        GAPoolingResult["simulationGAresults"]=case
        print (GAPoolingResult["GAP"])    
        results.append(GAPoolingResult)
            
        GAPoolingResult={}
            
with open('IJPR_Improved_GA_Clustering_v1.json', 'w') as outfile:
                json.dump(results, outfile)


#%%
json_case2=[]
with open("IJPR_Improved_GA_Clustering_v1.json", "r") as json_file:
    #json_file.readline()
    for line in json_file:
        json_case2.append(json.loads(line))

#for case in json_case:
#    print case["caseID"]

