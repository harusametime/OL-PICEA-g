import numpy as np
from gurobipy import *
import sys
from numpy import int
import pandas as pd
import random

from scipy.stats import norm


class problem:

    def __init__(self, l_eps, u_eps):


        np.random.seed(0)



        # Parameter of chance-constrained programming, probabity of holding
        self.alpha = 0.1

        price = np.loadtxt("./problem1/price.csv", delimiter=",")
        self.job = np.loadtxt("./problem1/job50.csv", delimiter=",")
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
#         for i in range(g_power_avg.shape[0]):
#             for j in range(g_power_avg.shape[1]):
#                 for t in range(self.n_sampling):
#                     if g_power_std[i,j] > 0:
#                         self.g_power[i,j,t] = np.random.normal(g_power_avg[i,j], g_power_std[i,j])
#                     else:
#                         self.g_power[i,j,t] = 0

      #  print self.g_power[:,:,0]
#
#         print g_power_std
#         print np.random.normal(g_power_avg, g_power_std)
# #        g_power_covar = np.diag(g_power_var)
# #         self.g_power = np.random.multivariate_normal(g_power_avg, g_power_covar, size=self.n_sampling)
#         self.g_power = np.reshape(self.g_power,[g_power_shape[0], g_power_shape[1], self.n_sampling])


        # Required poer
        # In our observation, each core needs 20W.
        self.r_power = self.job[:,0]
        self.r_power = self.r_power * 20 / 1000

#         deadline_range = np.array([self.job[:,1], self.n_slot - self.job[:,1]])
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
        n_population = 100
        max_gen = 100
        n_goals = 100
        obl_prob = 0.2  # we consider obl at rate 0.2
        f1_goal_bounds = [0, 50] # goal vectors have values of rand(0,50) for f1 and d(0,50) for f2
        f2_goal_bounds = [0, 50] # goal vectors have values of rand(0,50) for f1 and d(0,50) for f2


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
        #self.x = np.random.randint(2, size=(self.n_job, self.n_dc, self.n_slot))

        j = 0
        infeasible_counter = 0;
        while j < self.n_job:
            dc_select = random.randint(0, self.n_dc-1)
            slot_select = random.randint(0, self.n_slot - self.job[i,1]) # This ensures the duration self.job[i,1]

            # assign job from starting time indicated by "slot_select"
            for t in range(slot_select + self.job[i,1]-1):
                self.x[j,dc_select,t] = 1

            # Once a job is assigned, feasiblity of the solution is checked.
            # If feasible, go to the next job.
            if is_feasible(self.x):
                infeasible_counter = 0
                j += 1
            else:
                infeasible_counter += 1
                # If it is hard to get feasible solution, restart to generate solutions.
                if infeasible_counter > 10:
                    j = 0



    '''
    We have two objective functions. f_1 is procurement of power, and f_2 is total delay.
    For epsilon constraint method, we use only f_2 as the objective function and f_1 as
    a constraint.
    '''
    def _set_objective(self, u_eps):


        eps_expr = LinExpr()
        eps_expr.add(sum( self.r_power[i] * self.x[i,j,t] - self.y[i,j, t] for i in range(self.n_job) for j in range(self.n_dc) for t in range(self.n_slot)))
        self.model.setObjective(sum(self.d[i] for i in range(self.n_job)) - 0.001 * (u_eps - eps_expr), GRB.MINIMIZE)

        #self.model.setObjective(sum(self.d[i] for i in range(self.n_job)), GRB.MINIMIZE)
        #self.model.setObjective(sum( self.r_power[i] * self.x[i,j,t] - self.y[i,j, t] for i in range(self.n_job) for j in range(self.n_dc) for t in range(self.n_slot)), GRB.MINIMIZE)

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

    def is_feasibile(self):

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
                # constraint 1 & 2
                total_power = sum(self.r_power[i] * self.x[i,j,t] for i in range(self.n_job))
                if total_power > self.g_power_percentile[j,t]:
                    return False

                # constraint 3
                total_proc = sum(self.job[i,0] * self.x[i,j,t] for i in range(self.n_job))
                if total_proc > self.free_proc[j]:
                    return False



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