import multiprocessing
import random
import tqdm
import numpy as np
from itertools import combinations,product
import time
import logging

#%%
# Test your jupyter code here

#%%
 
MAX_CACHE_LEN = 10000 # max number of representations to cache
multipliers=None
cache=[]

def set_multipliers(len_sku):
    global multipliers
    multipliers = 2**np.arange(len_sku-1,-1,-1)

def ns_throas_mutation(x, *args):
    """ Randomly select three element positions(i,j,k) of x. 
         value at i becomes value at j 
         value at j becomes value at k
         value at k becomes value at i """
    x_new = x[:]
    n_skus = len(x_new)
    idx1, idx2, idx3 = random.sample(range(0, n_skus), 3)
    x_new[idx2] = x[idx1]
    x_new[idx3] = x[idx2]
    x_new[idx1] = x[idx3]   
    return x_new  

def ns_center_inverse_mutation(x, *args):
    """ Randomly select a position i, mirror genes referencing i 
    Example: [1,2,3,4,5,6] if i is 3 result is [3,2,1,6,5,4]"""
    idx = random.randint(0, len(x)-1)
    return  x[idx::-1] + x[:idx:-1]   


def ns_two_way_swap(x, *args):
    """ Randomly swaps two elements of x. """
    x_new = x[:]
    n_skus = len(x_new)
    idx1, idx2 = random.sample(range(0, n_skus), 2)
    x_new[idx1] = x[idx2]
    x_new[idx2] = x[idx1]
    return x_new


def ns_shuffle(x, *args):
    """ Returns a permutation of elements of x. """
    x_new = x[:]
    random.shuffle(x_new)
    return x_new


def ns_mutate_random(x, min_cluster, max_cluster):
    """ Changes value of a random element of x. 
        The new values are between min_cluster and max_cluster inclusive. """
    x_new = x[:]
    n_skus = len(x_new)
    idx = random.randint(0, n_skus-1)
    ex_cluster_number = x[idx]
    numbers = list(range(min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, max_cluster))
    x_new[idx] = random.choice(numbers)
    return x_new #if filter_out_symmetric_solutions([x_new]) else ns_mutate_random(x, min_cluster, max_cluster)


def ns_mutate_random2(x, min_cluster, max_cluster):
    """ Changes values of two random elements of x.
        The new values are between min_cluster and max_cluster inclusive. """
    x_new = x[:]
    n_skus = len(x_new)
    idx1, idx2 = random.sample(range(0, n_skus), 2)
    ex_cluster_number = x[idx1]
    numbers = list(range(min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, max_cluster))
    x_new[idx1] = random.choice(numbers)
    ex_cluster_number = x[idx2]
    numbers = list(range(min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, max_cluster))
    x_new[idx2] = random.choice(numbers)
    
    return x_new #if filter_out_symmetric_solutions([x_new]) else ns_mutate_random2(x, min_cluster, max_cluster)


def encode(x):
    """ 
    Encode x as a sorted list of base10 representation to detect symmetries.
        
    First, for each cluster determine which SKU types are available.
        x = [1,1,2] means in cluster 1 there are SKU1 and SKU2 in cluster 2 there is only SKU3
        So the binary representation(skill-server assignment) is below. 
        Each inner list represents a server and binary values denote if that server has the skill or not. 
        
        Examples of solutions that are symmetric to x, their binary and base 10 represetations:
               x     binary representation   base10 representation
            -------  ---------------------    ------------------
            [1,1,2]  [[1,1,0], [0,0,1]]             [6,1]
            [1,1,3]  [[1,1,0], [0,0,1]]             [6,1]
            [2,2,1]  [[0,0,1], [1,1,0]]             [1,6]
            [2,2,3]  [[1,1,0], [0,0,1]]             [6,1]
            [3,3,1]  [[0,0,1], [1,1,0]]             [1,6]
            [3,3,2]  [[0,0,1], [1,1,0]]             [1,6]
    :param x, sample solution
    """
    return sorted([sum([multipliers[j] for j,c2 in enumerate(x) if c2 == c]) for c in set(x)])    


def symmetry_control_pass(x):
    x_rep = encode(x)
    if x_rep not in cache:
        # if cache is full remove oldest added item
        if len(cache) == MAX_CACHE_LEN:
            cache.pop(0)
        cache.append(x_rep)
        return True
    return False


def filter_out_symmetric_solutions(candidates):
    # filter the ones causing symmetry
    filtered=[]
    start_time = time.time()
    for c in candidates:
        c_rep = encode(c)
        if c_rep not in cache:
            filtered.append(c)
            # if cache is full remove oldest added item
            if len(cache) == MAX_CACHE_LEN:
                cache.pop(0)
            cache.append(c_rep)
    stop_time = time.time() - start_time
    logging.debug(f"filtering:{stop_time} from {len(candidates)} to {len(filtered)}")
    return filtered


# neigbor list generation functions
def nlgf_one_gene_mutation(x):
    logging.debug(f"cache length: {len(cache)}")
    n_skus = len(x)
    start_time = time.time()
    # generate all possible mutations
    candidates = [x[:i] + [j] + x[i+1:] for i in range(n_skus) for j in range(1, n_skus+1) if j != x[i]]
    stop_time = time.time() - start_time
    logging.debug(f"generating one gene mutate neighbors:{stop_time}")
    # filter the ones causing symmetry
    filtered = filter_out_symmetric_solutions(candidates)
    return filtered


def get_indice_pairs_to_mutate(x):
    return combinations(range(len(x)), 2)


def nlgf_shuffle(x):
    pass


def nlgf_two_gene_mutation(x):
    logging.debug(f"cache length: {len(cache)}")
    start_time = time.time()
    n_sku = len(x)
    org_set = set(range(1, n_sku+1))
    candidates =[x[:idx1]+[val1]+x[idx1+1:idx2]+[val2]+x[idx2+1:] for idx1, idx2 in get_indice_pairs_to_mutate(x)
                                                              for val1, val2 in product(org_set.difference({x[idx1]}), org_set.difference({x[idx2]}))]
    stop_time = time.time() - start_time
    logging.debug(f"generating two gene mutate neighbors:{stop_time}")
    filtered = filter_out_symmetric_solutions(candidates)
    return filtered


def nlgf_two_way_swap(x):
    pass


def get_best_local_solution(solution_set, obj_func):
    pool = multiprocessing.Pool()
    fitness_vals = pool.map(obj_func, solution_set)
    tc_x_best = min(fitness_vals)
    best_idx = fitness_vals.index(tc_x_best)
    return solution_set[best_idx]


def solve_gvns(case_id, initial_solution, nsf, nlgf, obj_func, min_cluster, max_cluster, max_iters=500):
    """Finds best solution x given an initial solution x,
       list shaking functions nsf, and
       list of neighbor list generation functions  nlgf. """
    x = initial_solution
    global multipliers
    multipliers = 2**np.arange(len(x)-1,-1,-1)
    tcost_x = obj_func(x)[0]
    x_rep = encode(x)
    cache =[x_rep]
    total_search_time=0. 
    for epoch in tqdm.tqdm(range(max_iters),desc=f"{case_id}"):
        k = 0
        while k < len(nsf):
            # create neighborhood solution using kth ngf
            x1 = nsf[k](x, min_cluster, max_cluster)
            tcost_x1 = obj_func(x1)[0]
            l = 0
            while l < len(nlgf):
                logging.debug(f"epoch:{epoch+1} ngf:{k+1} nlgf:{l+1}")                  
                neighbor_set = nlgf[l](x1)
                # if no new neighbors after eliminating symmetries
                if not neighbor_set:
                    l += 1
                    continue
                start_time = time.time()
                x2 = get_best_local_solution(neighbor_set, obj_func)      
                stop_time = time.time() - start_time
                total_search_time += stop_time
                logging.debug(f"local search time:{stop_time} total search time:{total_search_time}")    
                tcost_x2 = obj_func(x2)[0]
                if tcost_x2 < tcost_x1:
                    if tcost_x2 < tcost_x:
                        logging.debug(f"===== New lower total cost from local search: {tcost_x2}")
                    x1 = x2
                    tcost_x1 = tcost_x2
                    l = 0
                else:
                    l += 1

            if tcost_x1 < tcost_x:
                x = x1
                tcost_x = tcost_x1
                k = 0
            else:
                k += 1
    return x


def solve_rvns(fname_possix, case_id, initial_solution, nsf, nlgf, obj_func, min_cluster, max_cluster, max_iters=1000):
    """Finds best solution x given an initial solution x,
       list shaking functions nsf, and
       list of neighbor list generation functions  nlgf. """
    logging.basicConfig(filename=f'results/app_rvns.log', level=logging.DEBUG, format=f'%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    x = initial_solution
    global multipliers, cache
    set_multipliers(len(x))
    tcost_x = obj_func(x)[0]
    x_rep = encode(x)
    cache =[x_rep]
    total_search_time=0.
    symmetry_loop_count=0 
    iter_since_last_best=0
    pbar = tqdm.tqdm(total = 250, desc=f"{case_id}")
    epoch=0
    while(iter_since_last_best < 250 ):
        k = 0
        epoch += 1
        better_found = False
        start_time = time.time()
        while k < len(nsf):
            # create neighborhood solution using kth ngf
            x1 = nsf[k](x, min_cluster, max_cluster)
            if not symmetry_control_pass(x1):
                symmetry_loop_count += 1
                if symmetry_loop_count > 500:
                    symmetry_loop_count = 0
                    k +=1
                continue
            else:
                symmetry_loop_count = 0
            tcost_x1 = obj_func(x1)[0]
            if tcost_x1 < tcost_x:
                logging.debug(f"neighbors:{fname_possix} {case_id} === NEW lower total cost: {tcost_x1:.4f} epoch{epoch} ===")
                x = x1
                tcost_x = tcost_x1
                k = 0
                better_found = True
            else:
                k += 1                
        
        # check for improvement
        if not better_found:
            iter_since_last_best += 1
            pbar.update(1)
        else:
            iter_since_last_best = 0
            pbar.close()
            pbar = tqdm.tqdm(total = 250, desc=f"{case_id}")

        stop_time = time.time() - start_time
        total_search_time += stop_time
        logging.debug(f"neighbors:{fname_possix} {case_id} epoch{epoch} time:{stop_time:.4f} total search time:{total_search_time:.4f} min_cost:{tcost_x}")  
    pbar.close()
    return x
