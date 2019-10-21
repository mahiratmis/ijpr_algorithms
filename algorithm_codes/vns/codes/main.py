from optimization_subroutine import evalOneMax, swicthtoOtherMutation,Final_evalOneMax
from vns import ns_mutate_random, ns_mutate_random2, ns_shuffle, ns_two_way_swap, \
    ns_throas_mutation, ns_center_inverse_mutation, \
    nlgf_one_gene_mutation, nlgf_two_gene_mutation, nlgf_shuffle, nlgf_two_way_swap, \
        solve_rvns, solve_gvns, encode

import numpy as np
import json
import time
import random
from deap import base
from deap import creator
from deap import tools
import sys
import os
import csv
import operator
import argparse
import itertools

def GAPoolingHeuristic(case_id, failure_rates, service_rates, holding_costs, penalty_cost, skill_cost, machine_cost, numSKUs, minCluster, maxCluster):

    # 1 is for maximization -1 for minimization
    # Minimize total cost
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def generateIndividual(numSKUs, minCluster, maxCluster):

        # Generating initial indvidual that are in the range of given max-min cluster numbers

        individual = [0]*numSKUs

        randomSKUsindex = np.random.choice(range(numSKUs), minCluster, replace=False)
        cluster_randomSKUs = np.random.choice(range(1, maxCluster+1), minCluster, replace=False)

        for i in range(minCluster):
            individual[randomSKUsindex[i]] = cluster_randomSKUs[i]

        for i in range(numSKUs):
            if individual[i] == 0:
                individual[i] = random.randint(1, maxCluster)

    # print type (creator.Individual(individual))
        return creator.Individual(individual)

    toolbox = base.Toolbox()

    # Attribute generator
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [1,number of SKUs] (i.e. 0 or 1 with equal
    #                      probability)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of #number of maximum cluster =#of SKUs 'attr_bool' elements ('genes')
    toolbox.register("individual", generateIndividual, numSKUs, minCluster, maxCluster)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # the goal ('fitness') function to be maximized
    # for objective function call pooling optimizer !!!
    # what values need for optimizer !!!

    # def evalOneMax(individual):
    #    return sum(individual),

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalOneMax, failure_rates, service_rates,
                     holding_costs, penalty_cost, skill_cost, machine_cost)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    #
    toolbox.register("mutate", swicthtoOtherMutation, indpb=0.4)
    # toolbox.register("mutate", swicthtoOtherMutation)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=10)

    # seed for reproducabiltiy
    random.seed(64)

    # create an initial population of 100 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=1)
    
    # initial solution
    x = [random.randint(1, numSKUs) for _ in range(numSKUs)]

    # Start Variable Neighborhood Search 
    x_best = solve_rvns(fname_possix, case_id, x, nsf, nlgf, toolbox.evaluate, minCluster, maxCluster)
    pop[0] = creator.Individual(x_best)
    TCs = list(map(toolbox.evaluate, [x_best]))
    for ind, tc in zip(pop, TCs):
        ind.fitness.values = tc
    best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (individual2cluster(best_ind), best_ind.fitness.values))
    return best_ind.fitness.values, x_best


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--neighbors', type=int, nargs="+", default=[2,3,4,5,6,1])
parser.add_argument('-mc', '--max_cases', type=int, default=128)
args = parser.parse_args()
indices= args.neighbors
max_cases = args.max_cases


# Read cases and their properties
file = "fullRangeResultsFullFlexNew.json"
json_case = [json.loads(line) for line in open(file, "r")]
sorted_assignments = sorted(json_case, key=operator.itemgetter('caseID'))
json_cases = sorted_assignments[:]

# shaking functions
nsf = [ns_mutate_random, ns_mutate_random2, ns_shuffle, ns_two_way_swap, ns_throas_mutation, ns_center_inverse_mutation]
# functions to generate list of local neighbors
nlgf = [nlgf_one_gene_mutation, nlgf_two_gene_mutation, nlgf_shuffle, nlgf_two_way_swap]

#for indices in [[3,4,5,6,1,2], [4,5,6,1,2,3],[3,1,2,4,5,6],[2,1,3,4,5,6]]:
#for indices in itertools.permutations(range(1,7),6):
#for indices in [[1,2,3,4,6],[2,3,4,5,1], [4,5,6,1,2,3],[3,1,2,4,5,6],[2,1,3,4,5,6]]:
print(indices)
fname_possix = "".join([str(i) for i in indices])

nsf =[nsf[i-1] for i in indices]
# nlgf = [nlgf[i-1] for i in indices]


# RUN of Algorithm STARTS HERE
# json_case
results = []
GAPoolingResult = {}
case_idx = 0

for case in json_cases:
    if case["caseID"] != "Case: 000x":
        failure_rates = case["failure_rates"]
        service_rates = case["service_rates"]
        holding_costs = case["holding_costs"]
        skill_cost = case["skill_cost"]
        penalty_cost = case["penalty_cost"]
        machine_cost = case["machine_cost"]

    if case_idx == max_cases:
        break
    print(f"{case['caseID']} started...")
    start_time = time.time()
    # unrestricted initial population _v4a
    numSKUs, minCluster, maxCluster = len(failure_rates), 1, len(failure_rates)
    _, best_ind = GAPoolingHeuristic(case["caseID"], np.array(failure_rates), np.array(service_rates),
                                    np.array(holding_costs), penalty_cost, skill_cost, machine_cost, numSKUs, minCluster, maxCluster)
    stop_time = time.time() - start_time
    # best individual is ran one more the for statistical data collection and recording
    # Using Final_evalOneMax
    bestCost, bestHolding, bestPenalty, bestMachineCost, bestSkillCost, bestCluster, bestS, bestEBO, bestserverAssignment = Final_evalOneMax(
        np.array(failure_rates), np.array(service_rates), np.array(holding_costs), penalty_cost, skill_cost, machine_cost, best_ind)
    GAPoolingResult["caseID"] = case["caseID"]
    GAPoolingResult["GAPoolingruntime"] = stop_time
    GAPoolingResult["GAPoolingTotalCost"] = bestCost
    GAPoolingResult["GAPoolingHoldingCost"] = bestHolding
    GAPoolingResult["GAPoolingPenaltyCost"] = bestPenalty
    GAPoolingResult["GAPoolingMachineCost"] = bestMachineCost
    GAPoolingResult["GAPoolingSkillCost"] = bestSkillCost
    GAPoolingResult["GAPoolingCluster"] = bestCluster
    GAPoolingResult["GAPoolingS"] = bestS
    GAPoolingResult["GAPoolingEBO"] = bestEBO
    GAPoolingResult["GAPoolingServerAssignment"] = bestserverAssignment
    # KmedianResult["KmedianLogFile"]=LogFileList

    GAPoolingResult["GAP"] = bestCost-case["total_cost"]
    GAPoolingResult["GAPoolingPercentGAP"] = 100 * \
        (bestCost-case["total_cost"])/case["total_cost"]

    GAPoolingResult["simulationGAresults"] = case
    results.append(GAPoolingResult)
    case_idx += 1
    with open(f'results/results{fname_possix}.csv', 'a') as csvfile:
        fieldnames = ['case_id', 'running_time', 'total_cost', 'holding_cost',
                    'penalty_cost', 'machine_cost', 'skill_cost', 'best_cluster',
                    'bestS', 'bestEBO', 'bestServerAssignment', 'GAP', 'GAPoolingPercentGAP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if case_idx == 1:
            writer.writeheader()
        writer.writerow({'case_id': case["caseID"],
                        'running_time': stop_time, 'total_cost': bestCost,
                        'holding_cost': bestHolding, 'penalty_cost': bestPenalty,
                        'machine_cost': bestMachineCost, 'skill_cost': bestSkillCost,
                        'best_cluster': bestCluster, 'bestS': bestS, 'bestEBO': bestEBO,
                        'bestServerAssignment': bestserverAssignment, 'GAP': GAPoolingResult["GAP"],
                        'GAPoolingPercentGAP': GAPoolingResult["GAPoolingPercentGAP"]})

    GAPoolingResult = {}

# Results are recorded as json file
with open(f'results/VNS_All{fname_possix}.json', 'w') as outfile:
    json.dump(results, outfile)

print("Program Finished Execution Succesfully.")
