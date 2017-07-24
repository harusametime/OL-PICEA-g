import numpy as np
import sys
from numpy import int
import pandas as pd
import random
import time

from sklearn.externals.joblib import Parallel, delayed
from scipy.stats import norm
from mpmath import rand
from ortools.constraint_solver._pywrapcp import new_BaseLns
from winnt import IMAGE_SCN_LNK_NRELOC_OVFL

'''
Evaluate optimality
''' 
def evaluate_optimality(obj1, obj2):
    current_eval = current_best.iloc[:, obj1:obj2+1].as_matrix()
    total_optimality = 0
    for e in range(true_eval.shape[0]):
        optimality = 0
        for c in range(current_eval.shape[0]):
            if true_eval[e, 0] == current_eval[c, 0]:
                optimality = 1 - float(current_eval[c, 1] - true_eval[e, 1]) / true_eval[e, 1]
                if optimality < 0:
                    optimality = 0
                print true_eval[e, 0], current_eval[c, 1], true_eval[e, 1], optimality
                break
            elif true_eval[e, 0] < current_eval[c, 0]:
                break
        
        total_optimality += optimality
    
    print "Average Optimality:", float(total_optimality) / true_eval.shape[0]
    print "Number of found solutions:", current_eval.shape[0]
'''
generate goal vectors
''' 
def generate_goals(population_with_eval, obj1, obj2, n_goals):
    obj1_range = np.array((population_with_eval.iloc[:,obj1].min(), population_with_eval.iloc[:,obj1].max()))
    obj2_range = np.array((population_with_eval.iloc[:,obj2].min(), population_with_eval.iloc[:,obj2].max()))
    obj1_goal_range = np.array((obj1_range[0], obj1_range[1]*1.2))
    obj2_goal_range = np.array((obj2_range[0], obj2_range[1]*1.2))          
    goals = np.empty((n_goals, 2))
    for g in range(n_goals):
        goals[g, 0] = np.random.rand() * (obj1_goal_range[1] - obj1_goal_range[0]) + obj1_goal_range[0]
        goals[g, 1] = np.random.rand() * (obj2_goal_range[1] - obj2_goal_range[0]) + obj2_goal_range[0]

    return goals
        
'''
update current_best
'''
def update_best(current_best, population, obj1, obj2):
    new_best= pd.DataFrame()
    current_best = current_best.append(population).sort_values(by=[obj1,obj2], ascending = True)
    min = 0
    max = 999999
    for key, row in current_best.iterrows():
        if row.iloc[obj1] > min and row.iloc[obj2] < max:
            _solution = np.reshape(row.values[:n_job * n_dc * n_slot], ( n_job, n_dc, n_slot ))
            if is_feasible(_solution, job, free_proc): 
                new_best = new_best.append(row)
                min = row.iloc[obj1]
                max = row.iloc[obj2]
                
    current_best = new_best
    return current_best
'''
Generate solutions by crossover.

'''
def crossover(sorted_population, n_crossover, parameters):

    _new_solutions = pd.DataFrame()

    #Crossover generates 2 solutions. We need  a half pairs of needed solutions.
    pairs = np.empty((n_crossover/2, 2),dtype=int)
    for i in range(n_crossover/2):
        pairs[i:] =np.array(random.sample(range(sorted_population.shape[0]), 2), dtype=int)
        
    solution_size= (n_job, n_dc, n_slot )
        
    _new_solutions = _new_solutions.append(Parallel(n_jobs=-1)([delayed(one_crossover)(sorted_population, solution_size, parameters, g_power_percentile, p) for p in pairs]))
    
    return _new_solutions


def one_crossover(sorted_population, solution_size, parameters, g_power_percentile, p):

    _new_solutions = pd.DataFrame()
    (job, free_proc, r_power, deadline) = parameters
    (n_job, n_dc, n_slot) = solution_size
    _solution = np.empty((len(p), n_job, n_dc, n_slot))

    for s in range(len(p)):
        _solution_in_panda = sorted_population.iloc[p[s], :].values[:n_job * n_dc * n_slot]
        _solution[s,:,:,:] = np.reshape(_solution_in_panda,  (n_job, n_dc, n_slot ))

    _new_solution = np.zeros((len(p), n_job, n_dc, n_slot))

    crossover_point = np.random.randint(0, n_dc -1 ) # return rand value in [0, n_dc-2]
    '''
    Now we consider crossover between solutions A and B in an element of pairs
    A_assigned (B_**) : index i,j,t when x_ijt = 1 in solution A (B)
    '''
    A_assigned = np.where(_solution[0,:,:,:]==1)
    B_assigned = np.where(_solution[1,:,:,:]==1)
    new_A_assigned = np.copy(A_assigned)
    new_B_assigned = np.copy(B_assigned)

    for a in range(A_assigned[0].shape[0]):
        # Crossover here
        if A_assigned[1][a] <= crossover_point:
            new_B_assigned[1][a] = A_assigned[1][a]
            new_B_assigned[2][a] = A_assigned[2][a]
        if B_assigned[1][a] > crossover_point:
            new_A_assigned[1][a] = B_assigned[1][a]
            new_A_assigned[2][a] = B_assigned[2][a]

        _new_solution[0,new_A_assigned[0][a],new_A_assigned[1][a],new_A_assigned[2][a]] = 1
        _new_solution[1,new_B_assigned[0][a],new_B_assigned[1][a],new_B_assigned[2][a]] = 1



    #print pd.DataFrame(np.append(np.reshape(_new_solution[0,...], -1), obj_values(_new_solution[0,...])))
    
    if is_feasible(_solution[0,...], job, free_proc):
        _new_solutions = _new_solutions.append(pd.DataFrame(np.append(np.reshape(_new_solution[0,...], -1), obj_values(_new_solution[0,...], parameters, g_power_percentile))).T)
    
    if is_feasible(_solution[1,...], job, free_proc):
        _new_solutions = _new_solutions.append(pd.DataFrame(np.append(np.reshape(_new_solution[1,...], -1), obj_values(_new_solution[1,...], parameters, g_power_percentile))).T)

    return _new_solutions

'''
Generate solutions by mutation

'''
def mutate(sorted_population, n_mutate):
    
    _new_solutions = pd.DataFrame()
    mutated_solution_index =np.array(random.sample(range(sorted_population.shape[0]), n_mutate), dtype=int)
    
    for m in mutated_solution_index:
        
        _solution_in_panda = sorted_population.iloc[m, :].values[:n_job * n_dc * n_slot]
        _solution = np.reshape(_solution_in_panda,  (n_job, n_dc, n_slot ))
        
        #which job assignment is mutated?
        mutated_j = np.random.randint(0, n_job)
        
        mutated_duration = job[mutated_j, 1]
        mutated_start = np.random.randint(0, n_slot - mutated_duration)
        mutated_dc = np.random.randint(0, n_dc)
         
        
        # list of 3 arrays of i, j ,t
        assigned = np.where(_solution==1)
        
        for a in range(assigned[0].shape[0]):
            if assigned[0][a] == mutated_j:
                for d in range(mutated_duration):
                    # decision variables become zero before the variables are mutated
                    _solution[assigned[0][a+d],assigned[1][a+d],assigned[2][a+d]] = 0
                    
                    # decision variables become one after mutation
                    assigned[1][a+d] = mutated_dc
                    assigned[2][a+d] = mutated_start + d
                
                _solution[assigned] = 1
                # Once the mutation on the variables for a job is finished, exit from the loop
                break
        
        if is_feasible(_solution, job, free_proc):
            _new_solutions = _new_solutions.append(pd.DataFrame(np.append(np.reshape(_solution, -1), obj_values(_solution, parameters, g_power_percentile))).T)
   
    return _new_solutions            
        

'''
This function generates initial solutions randomly.
However, random generation yields many infeasible solutions.
We assign jobs 1 and 13 that requires many CPU cores first.
If solution gets infeasible, this retries times of the specified numbers.
'''
def random_generate(x, retry =20):

    #x = np.zeros((n_job, n_dc, n_slot)

    j = 1
    dc_select = random.randint(2, 3)
    slot_select = random.randint(0, n_slot - job[j,1])
    for t in range(slot_select, slot_select + job[j,1]):
        x[j,dc_select,t] = 1

    j = 13
    dc_select = 4
    slot_select = random.randint(0, n_slot - job[j,1])
    for t in range(slot_select, slot_select + job[j,1]):
        x[j,dc_select,t] = 1

    j =0
    infeasible_counter = 0
    
    while j < n_job:
        if j == 1 or j ==13:
            j = j+1
            continue

        dc_select = random.randint(0, 4)
        slot_select = random.randint(0, n_slot - job[j,1]) # This ensures the duration job[i,1]

        # assign job from starting time indicated by "slot_select"

        for t in range(slot_select, slot_select + job[j,1]):
            x[j,dc_select,t] = 1

        # Once a job is assigned, feasiblity of the solution is checked.
        # If feasible, go to the next job.
        if  is_feasible(x, job, free_proc):
            infeasible_counter = 0
            j += 1
        else:
            for t in range(slot_select, slot_select + job[j,1]):
                x[j,dc_select,t] = 0

            infeasible_counter += 1
            # If it is hard to get feasible solution, restart to generate solutions.
            if infeasible_counter > retry:
                return False

    return True


'''
Because this program assigns job to hold the constraints:
- job is not interrupted
- job is completed within the slot,
then only the following constraints need to be checked.
- Job does not need more processors
'''
def is_feasible(x, job, free_proc):
    
    (n_job, n_dc, n_slot) = x.shape
    _solution = x

    for j in range(n_dc):
        for t in range(n_slot):
            total_proc =0
            # constraint 3
            for i in range(n_job):
                total_proc += job[i,0] * _solution[i,j,t]

            #total_proc = sum(job[i,0] * x[i,j,t] for i in range(n_job)
            
            if total_proc > free_proc[j]:
                return False

    return True

'''
Return values of two objective functions. total delay D and power P
'''
def obj_values(solution, parameters, g_power_percentile):

    (job, free_proc, r_power, deadline) = parameters
    D = P = 0
    (n_job, n_dc, n_slot) = solution.shape
    p_consumption = np.zeros((n_dc, n_slot))
    for j in range(n_job):

        # This returns
        # assignment[0]: data center to process the job (same value in the array)
        # assignment[1]: time slot for the job like t= 1, 2, 3, 4.
        assignment = np.where(solution[j,:,:] == 1)
        end_time = assignment[1][job[j,1]-1]
        if end_time > deadline[j]:
            D += end_time - deadline[j]

        for a in assignment[1]:
            p_consumption[assignment[0][0], a] += r_power[j]

    p_consumption = np.maximum(p_consumption-g_power_percentile, np.zeros((n_dc, n_slot)) )
    P = np.sum(p_consumption)

    return D, P


#            D = sum(model.getVarByName('d' + str(i)).x for i in range(n_job))
#            P = sum(r_power[i] * model.getVarByName('x' + str(i) +','+ str(j) +','+ str(t)).x - model.getVarByName('y' + str(i) +','+ str(j) +','+ str(t)).x for i in range(n_job) for j in range(n_dc) for t in range(n_slot))


    #model.setObjective(sum(d[i] for i in range(n_job)), GRB.MINIMIZE)
    #model.setObjective(sum( r_power[i] * x[i,j,t] - y[i,j, t] for i in range(n_job) for j in range(n_dc) for t in range(n_slot)), GRB.MINIMIZE)


'''
X is an array of vector x
We assume that each variable is in {0,1}
This return x_i = rand(0,1)-x_i.
'''
def opposite_vectors(X):
    new_X = np.random.randint(2, size=X.shape) - X
    index = np.where(new_X < 0)
    for s,i,j,t in zip(index[0],index[1],index[2],index[3]):
         new_X[s,i,j,t] = random.randint(0,1)
    return new_X


'''
show elapsed time
'''
def show_time():
    print ("elapsed_time:{0}".format(time.time() - start) + "[sec.]")
    
if __name__ == '__main__':

    '''
    flag_OL indicates that this uses Opposition based Learning strategy
    In our problem, this strategy generate infeasible solutions due to the constraint.
    As default, this flag is set to False.
    '''
    flag_OL = False

    np.random.seed(0)

    n_job = 25
    
    # True evaluation for comparison
    true_eval = np.loadtxt("../problem1/opteval_" + str(n_job) + "job.txt", delimiter=" ")
    true_eval = true_eval[:,1:3]
    
    
    # Parameter of chance-constrained programming, probabity of holding
    alpha = 0.1
    price = np.loadtxt("../problem1/price.csv", delimiter=",")
    job = np.loadtxt("../problem1/job50.csv", delimiter="," , dtype=int)
    job = job[:n_job, :]
    

    insolation = np.loadtxt("../problem1/insolation.csv",delimiter =",") # Unit of insolation is 0.01[MJ/m^2] in the file
    #insolation = insolation[:, :8]

    n_slot = insolation.shape[1]   # the number of time slots considered in scheduling (indexed by t)

    n_dc = insolation.shape[0]     # the number of data centers  (indexed by j)
    n_job = job.shape[0]           # the number of jobs (indexed by i)

    free_proc = np.array([64, 64, 128, 128, 256]) # the number of cores
    
    # Loss of photovoltaics
    loss = 0.7

    # 250w x 20 panels (about 30m^2, equivalent to one container)
    capacity = 5

    #sampline times
    #n_sampling = 50

    # generated power
    g_power_avg =  loss * 0.2778 * insolation *  0.01 * capacity
#         g_power_shape =  g_power_avg.shape
#         g_power_avg = g_power_avg.flatten()

    g_power_std = 0.1 * g_power_avg

    #g_power = np.empty([g_power_avg.shape[0],g_power_avg.shape[1], n_sampling])

    g_power_percentile = np.empty([g_power_avg.shape[0],g_power_avg.shape[1]])
    for j in range(g_power_avg.shape[0]):
        for t in range(g_power_avg.shape[1]):
            if g_power_avg[j,t] >0:
                g_power_percentile[j,t] = norm.ppf(q=alpha, loc=g_power_avg[j,t], scale=g_power_std[j,t])
            else:
                g_power_percentile[j,t] = 0
#
    # Required poer
    # In our observation, each core needs 20W.
    r_power = job[:,0]
    r_power = r_power * 20.0 / 1000.0

    # deadline_range = np.array([job[:,1], n_slot - job[:,1]])
    deadline = np.empty(n_job)
    for i in range(n_job):
        #print deadline_range[0, i],deadline_range[1, i]
        #if deadline_range[0, i] <  deadline_range[1, i]:
        if n_slot > job[i,1]:
            deadline[i] = job[i,1] + int(np.random.randint(n_slot - job[i,1] ))
        else:
            deadline[i] = job[i,1]
        #else:
        #    deadline[i] = job[i,1]  + np.random.randint(0, deadline_range[0, i])


    parameters = (job, free_proc, r_power, deadline)

    '''
    Parameters for PICEA-g

    '''
    mutation_rate = 0.1
    n_population = 100
    max_gen = 100
    n_goals = 100
    obl_prob = 0.2  # we consider obl at rate 0.2
    goal_bounds = np.array([[0, 50], [20, 50]]) # goal vectors have values of rand(0,50) for f1 and d(0,50) for f1
                                      # goal vectors have values of rand(20,50) for f1 and d(0,50) for f2
    '''
    x: decision variable (0: job is not assigned, 1: job is assigned)
    y: sub decision variable for g_ijt * x_ijt
    z: sub decision variable for judging whether the job is not interrupted
    d: sub decision variable for delay
    g: sub decision variable for power assigned for processing job i at slot t in DC j
    w: sub decision variable for sample average approximation
    '''

    # d = [ model.addVar(n_job, ub = n_slot, name = "d" +str(i)) for i in range(n_job)]

    
    
    start = time.time()


    '''
    Randomly generate initial solutions:
    S1: the half of them are randomly generated.
    S2: the rest of them are generated by Opposition based learning
    '''

    count = 0
    if flag_OL:
        population = np.zeros((n_population/2, n_job, n_dc, n_slot))
    else:
        population = np.zeros((n_population, n_job, n_dc, n_slot))

    if flag_OL:
        while count < n_population /2:
            x = np.zeros((n_job, n_dc, n_slot))
            if random_generate(x):
                generated_solution = x.copy()
                population[count,:,:,:] = generated_solution
                count +=1
            else:
                print "error"
                continue
        op_population = opposite_vectors(population)
        population = np.concatenate((population, op_population),axis=0)
    else:
        while count < n_population:
            x = np.zeros((n_job, n_dc, n_slot))
            if random_generate(x):
                generated_solution = x.copy()
                population[count,:,:,:] = generated_solution
                count +=1
            else:
                print "error"
                continue

    
    # evaluation  values are stored in this list
    # each population has two objective function values
    population_with_eval = []
    #eval_list = np.empty((n_population, 3))

    for i in range(n_population):
        evaluation = obj_values(population[i,:,:,:], parameters, g_power_percentile)
        flatten_solution = np.reshape(population[i], -1)
        population_with_eval.append(np.concatenate((flatten_solution,evaluation)))
        #evaluation = obj_values(population[i,:,:,:])
        #population_with_eval[i,:,:,:,0,:] = evaluation[0]
        #population_with_eval[i,:,:,:,:,0] =  evaluation[1]



    '''
    Now sorted_population is a vector of which size is n_job * n_dc * n_slot + 2 eval_values
    For example, 30 jobs, 5 DCs  and 24 slots yields 3600 values for a solution. The solution
    can be converted by reshape(first3600, (30, 5, 24)).
    '''
    population_with_eval = pd.DataFrame(population_with_eval)
    n_columns = population_with_eval.shape[1]
    obj1 = n_columns-2
    obj2 = n_columns-1
    current_best = pd.DataFrame()
    current_best = update_best(current_best, population_with_eval, obj1, obj2)


    print current_best
    '''
    Generate goal vectors for 2 objectives
    '''
    goals = generate_goals(population_with_eval, obj1, obj2, n_goals)


    '''
    loop for evolutionary algorithm
    '''
    itr_generation = 1
    while True:
        '''
        Generate next solutions
        '''
        print "##", itr_generation, " generation"
        itr_generation = itr_generation+ 1
        
        n_crossover = int(n_population * (1-mutation_rate))
        parameter = (job, free_proc)
        cross_over_population = crossover(population_with_eval, n_crossover, parameters)
        mutate_population = mutate(population_with_eval, n_population - n_crossover)
        
        population_with_eval = population_with_eval.append(cross_over_population)
        population_with_eval = population_with_eval.append(mutate_population)
         
        current_best = update_best(current_best, population_with_eval, obj1, obj2)
        
        
        '''
        Evaluation summary
        - optimality
        - found candidate pareto optimal solutions
        - computational time  
        '''
        evaluate_optimality(obj1, obj2)
        
        show_time()
        
        
        
        '''
        add goal vectors
        '''
        goals = np.vstack((goals, generate_goals(population_with_eval, obj1, obj2, n_goals)))
        
        '''
        Calculate fitness
        '''
        satisfied_number = np.zeros((goals.shape[0]))
        solution_fitness = np.zeros((len(population_with_eval)))
        goal_fitness = np.zeros((goals.shape[0],1))
        
        for n in range(goals.shape[0]):
            for p in range(len(population_with_eval)):
                if goals[n, 0] >= population_with_eval.iloc[p,obj1] and goals[n, 1] >= population_with_eval.iloc[p,obj2]:
                    satisfied_number[n] += 1
        
        for p in range(len(population_with_eval)):
            for n in range(goals.shape[0]):
                if goals[n, 0] >= population_with_eval.iloc[p,obj1] and goals[n, 1] >= population_with_eval.iloc[p,obj2]:
                    solution_fitness[p] += float(1/satisfied_number[n])
        
        goal_fitness = (satisfied_number-1) / (2 * n_goals - 1)
        goal_fitness[goal_fitness < 0] = 0
        goal_fitness = 1 + (1+ goal_fitness)
        
        
        '''
        select solutions and goals based on fitness
        '''
        temp = (-solution_fitness).argsort()
        ranks = np.empty(len(solution_fitness), int)
        ranks[temp] = np.arange(len(solution_fitness))
        population_with_eval = population_with_eval.iloc[ranks < n_goals]
     
        temp = (-goal_fitness).argsort()
        ranks = np.empty(len(goal_fitness), int)
        ranks[temp] = np.arange(len(goal_fitness))
        goals = goals[ranks < n_goals]
                    

   