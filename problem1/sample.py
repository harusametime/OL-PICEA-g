import numpy as np
import sys
from numpy import int
import pandas as pd
import random

from scipy.stats import norm
from mpmath import rand
from ortools.constraint_solver._pywrapcp import new_BaseLns


class problem:

    def __init__(self, l_eps, u_eps):


        '''
        flag_OL indicates that this uses Opposition based Learning strategy
        In our problem, this strategy generate infeasible solutions due to the constraint.
        As default, this flag is set to False.

        '''
        flag_OL = False

        np.random.seed(0)

        # Parameter of chance-constrained programming, probabity of holding
        self.alpha = 0.1

        price = np.loadtxt("./problem1/price.csv", delimiter=",")
        self.job = np.loadtxt("./problem1/job50.csv", delimiter="," , dtype=int)
        self.job = self.job[:30, :]

        insolation = np.loadtxt("./problem1/insolation.csv",delimiter =",") # Unit of insolation is 0.01[MJ/m^2] in the file
        #insolation = insolation[:, :8]

        self.n_slot = insolation.shape[1]   # the number of time slots considered in scheduling (indexed by t)

        self.n_dc = insolation.shape[0]     # the number of data centers  (indexed by j)
        self.n_job = self.job.shape[0]           # the number of jobs (indexed by i)

        self.free_proc = np.array([64, 64, 128, 128, 256]) # the number of cores

        # Loss of photovoltaics
        loss = 0.7

        # 250w x 20 panels (about 30m^2, equivalent to one container)
        capacity = 5

        #sampline times
        #self.n_sampling = 50

        # generated power
        g_power_avg =  loss * 0.2778 * insolation *  0.01 * capacity
#         g_power_shape =  g_power_avg.shape
#         g_power_avg = g_power_avg.flatten()

        g_power_std = 0.1 * g_power_avg

        #self.g_power = np.empty([g_power_avg.shape[0],g_power_avg.shape[1], self.n_sampling])

        self.g_power_percentile = np.empty([g_power_avg.shape[0],g_power_avg.shape[1]])
        for j in range(g_power_avg.shape[0]):
            for t in range(g_power_avg.shape[1]):
                if g_power_avg[j,t] >0:
                    self.g_power_percentile[j,t] = norm.ppf(q=self.alpha, loc=g_power_avg[j,t], scale=g_power_std[j,t])
                else:
                    self.g_power_percentile[j,t] = 0
#
        # Required poer
        # In our observation, each core needs 20W.
        self.r_power = self.job[:,0]
        self.r_power = self.r_power * 20.0 / 1000.0

        # deadline_range = np.array([self.job[:,1], self.n_slot - self.job[:,1]])
        self.deadline = np.empty(self.n_job)
        for i in range(self.n_job):
            #print deadline_range[0, i],deadline_range[1, i]
            #if deadline_range[0, i] <  deadline_range[1, i]:
            if self.n_slot > self.job[i,1]:
                self.deadline[i] = self.job[i,1] + int(np.random.randint(self.n_slot - self.job[i,1] ))
            else:
                self.deadline[i] = self.job[i,1]
            #else:
            #    self.deadline[i] = self.job[i,1]  + np.random.randint(0, deadline_range[0, i])



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

        # self.d = [ self.model.addVar(self.n_job, ub = self.n_slot, name = "d" +str(i)) for i in range(self.n_job)]

        self.x = np.zeros((self.n_job, self.n_dc, self.n_slot))
        self.y = np.zeros((self.n_job, self.n_dc, self.n_slot))
        self.z = np.zeros((self.n_job, self.n_dc, self.n_slot))
        self.d = np.zeros((self.n_job, self.n_dc, self.n_slot))
        self.g = np.zeros((self.n_job, self.n_dc, self.n_slot))
        #self.w ={}


        '''
        Randomly generate initial solutions:
        S1: the half of them are randomly generated.
        S2: the rest of them are generated by Opposition based learning
        '''

        count = 0
        if flag_OL:
            population = np.zeros((n_population/2, self.n_job, self.n_dc, self.n_slot))
        else:
            population = np.zeros((n_population, self.n_job, self.n_dc, self.n_slot))

        if flag_OL:
            while count < n_population /2:
                if self.random_generate():
                    generated_solution = self.x.copy()
                    population[count,:,:,:] = generated_solution
                    count +=1
                else:
                    print "error"
                    continue
            op_population = self.opposite_vectors(population)
            population = np.concatenate((population, op_population),axis=0)
        else:
            while count < n_population:
                if self.random_generate():
                    generated_solution = self.x.copy()
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
            evaluation = self.obj_values(population[i,:,:,:])
            flatten_solution = np.reshape(population[i], -1)
            population_with_eval.append(np.concatenate((flatten_solution,evaluation)))
            #evaluation = self.obj_values(population[i,:,:,:])
            #population_with_eval[i,:,:,:,0,:] = evaluation[0]
            #population_with_eval[i,:,:,:,:,0] =  evaluation[1]



        population_with_eval = pd.DataFrame(population_with_eval)
        n_columns = population_with_eval.shape[1]
        obj1 = n_columns-2
        obj2 = n_columns-1
        sorted_population =  population_with_eval.sort_values(by=[obj1,obj2], ascending = True)

        '''
        Now sorted_population is a vector of which size is n_job * n_dc * n_slot + 2 eval_values
        For example, 30 jobs, 5 DCs  and 24 slots yields 3600 values for a solution. The solution
        can be converted by reshape(first3600, (30, 5, 24)).
        '''
        current_best = pd.DataFrame()
        min = 0
        for key, row in sorted_population.iterrows():
            if row.iloc[obj1] > min:
                _solution = np.reshape(row.values[:self.n_job * self.n_dc * self.n_slot], ( self.n_job, self.n_dc, self.n_slot ))
                if self.is_feasible(_solution):
                    current_best.append(row)
                    min = row.iloc[obj1]

        '''
        Geenrate goal vectors for 2 objectives
        '''
        goals = np.empty((n_goals, 2))
        for g in range(2):
            goals[:,g] = np.random.rand(n_goals) * (goal_bounds[g,1]-goal_bounds[g,0]) + goal_bounds[g,0]


        '''
        loop for evolutionary algorithm
        '''
        while True:
            '''
            Generate next solutions
            '''
            #generated by cross over
            n_crossover = int(n_population * (1-mutation_rate))
            new_population = self.crossover(sorted_population, n_crossover)


            break


    '''
    Generate solutions by crossover.

    '''
    def crossover(self,sorted_population, n_crossover):

        _new_solutions = pd.DataFrame()

        #Crossover generates 2 solutions. We need  a half pairs of needed solutions.
        pairs = np.empty((n_crossover/2, 2),dtype=int)
        for i in range(n_crossover/2):
            pairs[i:] =np.array(random.sample(range(sorted_population.shape[0]), 2), dtype=int)

        for p in pairs:
            _solution = np.empty((len(p), self.n_job, self.n_dc, self.n_slot))

            for s in range(len(p)):
                _solution_in_panda = sorted_population.iloc[p[s], :].values[:self.n_job * self.n_dc * self.n_slot]
                _solution[s,:,:,:] = np.reshape(_solution_in_panda,  (self.n_job, self.n_dc, self.n_slot ))

            _new_solution = np.zeros((len(p), self.n_job, self.n_dc, self.n_slot))

            crossover_point = np.random.randint(0, self.n_dc -1 ) # return rand value in [0, self.n_dc-2]
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
                if B_assigned[1][a] > crossover_point:
                    new_A_assigned[1][a] = B_assigned[1][a]

                _new_solution[0,new_A_assigned[0][a],new_A_assigned[1][a],new_A_assigned[2][a]] = 1
                _new_solution[1,new_B_assigned[0][a],new_B_assigned[1][a],new_B_assigned[2][a]] = 1



            #print pd.DataFrame(np.append(np.reshape(_new_solution[0,...], -1), self.obj_values(_new_solution[0,...])))
            _new_solutions = _new_solutions.append(pd.DataFrame(np.append(np.reshape(_new_solution[0,...], -1), self.obj_values(_new_solution[0,...]))).T)
            _new_solutions = _new_solutions.append(pd.DataFrame(np.append(np.reshape(_new_solution[1,...], -1), self.obj_values(_new_solution[1,...]))).T)

        return _new_solutions

    '''
    This function generates initial solutions randomly.
    However, random generation yields many infeasible solutions.
    We assign jobs 1 and 13 that requires many CPU cores first.
    If solution gets infeasible, this retries times of the specified numbers.
    '''
    def random_generate(self,retry =20):

        self.x = np.zeros((self.n_job, self.n_dc, self.n_slot))

        j = 1
        dc_select = random.randint(2, 3)
        slot_select = random.randint(0, self.n_slot - self.job[j,1])
        for t in range(slot_select, slot_select + self.job[j,1]):
            self.x[j,dc_select,t] = 1

        j = 13
        dc_select = 4
        slot_select = random.randint(0, self.n_slot - self.job[j,1])
        for t in range(slot_select, slot_select + self.job[j,1]):
            self.x[j,dc_select,t] = 1

        j =0
        infeasible_counter = 0
        while j < self.n_job:
            if j == 1 or j ==13:
                j = j+1
                continue

            dc_select = random.randint(0, 4)
            slot_select = random.randint(0, self.n_slot - self.job[j,1]) # This ensures the duration self.job[i,1]

            # assign job from starting time indicated by "slot_select"

            for t in range(slot_select, slot_select + self.job[j,1]):
                self.x[j,dc_select,t] = 1

            # Once a job is assigned, feasiblity of the solution is checked.
            # If feasible, go to the next job.
            if  self.is_feasible():
                infeasible_counter = 0
                j += 1
            else:
                for t in range(slot_select, slot_select + self.job[j,1]):
                    self.x[j,dc_select,t] = 0

                infeasible_counter += 1
                # If it is hard to get feasible solution, restart to generate solutions.
                if infeasible_counter > retry:
                    return False

        return True


    '''
    Because this program assigns job to hold the constraints:
    - job is not interrupted
    - job is completed wihtin the slot,
    then only the following constraints need to be checked.
    - Job does not need more processors
    '''
    def is_feasible(self, _solution = None):
        if _solution is None:
            _solution = self.x

        for j in range(self.n_dc):
            for t in range(self.n_slot):
                total_proc =0
                # constraint 3
                for i in range(self.n_job):
                    total_proc += self.job[i,0] * _solution[i,j,t]

                #total_proc = sum(self.job[i,0] * self.x[i,j,t] for i in range(self.n_job))
                if total_proc > self.free_proc[j]:
                    return False

        return True

    '''
    Return values of two objective functions. total delay D and power P
    '''
    def obj_values(self, solution):

        D = P = 0

        p_consumption = np.zeros((self.n_dc, self.n_slot))
        for j in range(self.n_job):

            # This returns
            # assignment[0]: data center to process the job (same value in the array)
            # assignment[1]: time slot for the job like t= 1, 2, 3, 4.
            assignment = np.where(solution[j,:,:] == 1)

            end_time = assignment[1][self.job[j,1]-1]
            if end_time > self.deadline[j]:
                D += end_time - self.deadline[j]

            for a in assignment[1]:
                p_consumption[assignment[0][0], a] += self.r_power[j]

        p_consumption = np.maximum(p_consumption-self.g_power_percentile, np.zeros((self.n_dc, self.n_slot)) )
        P = np.sum(p_consumption)

        return D, P


#            D = sum(self.model.getVarByName('d' + str(i)).x for i in range(self.n_job))
#            P = sum(self.r_power[i] * self.model.getVarByName('x' + str(i) +','+ str(j) +','+ str(t)).x - self.model.getVarByName('y' + str(i) +','+ str(j) +','+ str(t)).x for i in range(self.n_job) for j in range(self.n_dc) for t in range(self.n_slot))


        #self.model.setObjective(sum(self.d[i] for i in range(self.n_job)), GRB.MINIMIZE)
        #self.model.setObjective(sum( self.r_power[i] * self.x[i,j,t] - self.y[i,j, t] for i in range(self.n_job) for j in range(self.n_dc) for t in range(self.n_slot)), GRB.MINIMIZE)


    '''
    X is an array of vector x
    We assume that each variable is in {0,1}
    This return x_i = rand(0,1)-x_i.
    '''
    def opposite_vectors(self,X):
        new_X = np.random.randint(2, size=X.shape) - X
        index = np.where(new_X < 0)
        for s,i,j,t in zip(index[0],index[1],index[2],index[3]):
             new_X[s,i,j,t] = random.randint(0,1)
        return new_X

    '''
    Basic epsion constraint method uses epsilon value as the upper bound on f_1,
    but our method uses upper (u_eps) and lower bounds (l_eps) on fq_1.
    '''
    def _set_eps_constraint(self, l_eps, u_eps):

        eps_expr = LinExpr()
        eps_expr.add(sum( self.r_power[i] * self.x[i,j,t] - self.y[i,j, t] for i in range(self.n_job) for j in range(self.n_dc) for t in range(self.n_slot)))
        #eps_expr.add(sum(self.d[i] for i in range(self.n_job)))
        self.u_eps_constr = self.model.addConstr(eps_expr <= u_eps, "u_eps_const")
        self.l_eps_constr = self.model.addConstr(eps_expr >= l_eps, "l_eps_const")

    def _set_constraints(self):

        '''
        Constraint1: total consumped power can not be more than generated power (sample average approximation
        Constraint2: total consumped power can not be more than total required power
        Constraint3: the number of cores used for processing can not be more than the number of free processors
        constraint4: sum of x must be equal to L
        Constraint5: job processing can not be interrupted
        Constraint6: subconstraint for constraint 5(Start point of job processing is only one)
        Constraint7: count of w_a must not be less than S(1-alpha)
        Constraint8: sum of z must be equal to 1
        '''

        for j in range(self.n_dc):
            for t in range(self.n_slot):
                const1_expr = LinExpr()
                const1_expr.add(sum(self.y[i,j,t] for i in range(self.n_job)))
                self.model.addConstr(const1_expr <= self.g_power_percentile[j,t])
                #print self.g_power_percentile[j,t]



        for j in range(self.n_dc):
            for t in range(self.n_slot):
                const3_expr = LinExpr()

                for i in range(self.n_job):
                    # constraint 2
                    self.model.addConstr(self.y[i,j,t] <= self.r_power[i] * self.x[i,j,t])

                    # sum of x[i,j,t] for const3
                    const3_expr.add(self.x[i,j,t], self.job[i,0])

                # const3
                self.model.addConstr(const3_expr <= self.free_proc[j])

            for i in range(self.n_job):
                L = int(self.job[i, 1])   # Length of job


                for t in range(self.n_slot):


                    self.model.addConstr(self.d[i]-t*self.z[i,j,t] >= self.job[i,1]-self.deadline[i]-1)

                    if t+ L > self.n_slot:
                        self.model.addConstr(self.z[i,j,t]== 0)

                    else:
                        #constraint5
                        const_test = LinExpr()
                        # for l in range(t,t+L):
                        #     print 'x['+str(i)+','+str(j)+','+str(l)+']+',
                        #     const_test.add(self.x[i,j,l])
                        #
                        # print '=z['+str(i)+','+str(j)+','+str(t)+']*',
                        # print L
                        # self.model.addConstr(const_test>= self.z[i,j,t] * L)

                        self.model.addConstr(sum(self.x[i,j,l] for l in range(t,t+L)) >= self.z[i,j,t] * L)


        for  i in range(self.n_job):
            L = int(self.job[i, 1])   # Length of job
            const4_expr = LinExpr()
            const8_expr = LinExpr()
            for j in range(self.n_dc):

                for t in range(self.n_slot):
                    const4_expr.add(self.x[i,j,t])

                    if t < self.n_slot:

                        const8_expr.add(self.z[i,j,t])




            self.model.addConstr(const4_expr == L)   #const4
            self.model.addConstr(const8_expr == 1 )   #const8

        self.model.update()


    def update_objfunc(self, u_eps):
        self._set_objective(u_eps)
        self.model.update()


    def update_eps_constraint(self, u_eps):
        self.u_eps_constr.setAttr(GRB.Attr.RHS, u_eps)
        self.model.update()

    def update_bound(self, deadline):
        self.model.setAttr("ObjBound", deadline)

    def solve(self, verbose = 1):

        if verbose == 1:
            self.model.setParam('OutputFlag', 0)

        try:
            self.model.optimize()
            # for v in self.model.getVars():
            #     if 'x' in v.varName and v.x ==1:
            #         print ('%s %g' % (v.varName, v.x))
            #     if 'z' in v.varName and v.x ==1:
            #         print ('%s %g' % (v.varName, v.x))


#             P = self.model.objVal
#             D = sum(self.model.getVarByName('d' + str(i)).x for i in range(self.n_job))

            D = sum(self.model.getVarByName('d' + str(i)).x for i in range(self.n_job))
            P = sum(self.r_power[i] * self.model.getVarByName('x' + str(i) +','+ str(j) +','+ str(t)).x - self.model.getVarByName('y' + str(i) +','+ str(j) +','+ str(t)).x for i in range(self.n_job) for j in range(self.n_dc) for t in range(self.n_slot))

            return D, P

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')
#        X= pd.DataFrame(index=range(self.n_dc), columns=range(self.n_slot)).fillna(0)
#        pd.set_option("display.max_columns", 80)
#
#        D = 0   #delay
#        P = 0   #Procurement
#
#        result_status = self.solver.Solve()
#
#        # The problem has an optimal solution.
#        if result_status != pywraplp.Solver.OPTIMAL:
#            return X, P, D, False
##        assert result_status == pywraplp.Solver.OPTIMAL
#
#        # The solution looks legit (when using self.solvers other than
#        # GLOP_LINEAR_PROGRAMMING, verifying the solution is highly recommended!).
#        assert self.solver.VerifySolution(1e-7, True)
#
#        if verbose == 2:
#            print'Number of variables =', self.solver.NumVariables()
#            print'Number of constraints =', self.solver.NumConstraints()
#
#            # The objective value of the solution.
#            print 'Optimal objective value = %d' % self.solver.Objective().Value()
#
##         if verbose >=1 :
##             for i in range(self.n_job):
##                 for j in range(self.n_dc):
##                     for t in range(self.n_slot):
##                         x_name = 'x' + str(i) +','+ str(j) +','+ str(t)
##                         print x_name, "=",
##                         print self.solver.LookupVariable(x_name).solution_value()
#
#        #Solutions
#        # (note that job at the same time in the same DC is overwritten)
#
#
#        #print X
#
#
#
#        for i in range(self.n_job):
#            d_name = 'd' + str(i)
#            D += self.solver.LookupVariable(d_name).solution_value()
#            for j in range(self.n_dc):
#                for t in range(self.n_slot):
#                    x_name = 'x' + str(i) +','+ str(j) +','+ str(t)
#                    if self.solver.LookupVariable(x_name).solution_value() == 1:
#                        if X.loc[j,t] == 0:
#                            X.loc[j,t] = str(i+1)
#                        else:
#
#                            temp = X.loc[j,t]
#                            temp += '/' + str(i+1)
#                            X.loc[j,t] = temp
#
#
#        for i in range(self.n_job):
#            for j in range(self.n_dc):
#                for t in range(self.n_slot):
#
#                    x_name ='x' + str(i) +','+ str(j) +','+ str(t)
#                    #self.eps_constraint.SetCoefficient(self.solver.LookupVariable(x_name), self.r_power[i])
#                    y_name ='y' + str(i) +','+ str(j) +','+ str(t)
#                    #self.eps_constraint.SetCoefficient(self.solver.LookupVariable(y_name), -1)
#                    P += self.r_power[i] * self.solver.LookupVariable(x_name).solution_value() - self.solver.LookupVariable(y_name).solution_value()
#
#
#        return X, D, P, True
#
#    def _inf_format(self,eps):
#        if eps == float("inf"):
#            return self.solver.infinity()
#        elif eps == float("-inf"):
#            return -self.solver.infinity()
#        else:
#            return eps
