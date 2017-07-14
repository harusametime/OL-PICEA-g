import problem1.sample as sample
import numpy as np
import math
import time
import sys
from multiprocessing import Process, Queue, Value, Array
import pandas as pd


if __name__ == '__main__':


    epsilon = 1000
    p = sample.problem(0,epsilon)



    '''
        X: Solution (Assignment of  jobs to slots), Panda DataFrame
        D: Deadline violation (\in R)
        P: Power Procurement (\in R)
        f: feasibility of the problem (True or False)
    '''
    D, P = p.solve(verbose =0)

    print "(D, P) = ", D, ",", P

    while True:
        epsilon = P - 0.1
        print epsilon
        p.update_objfunc(epsilon)
        p.update_eps_constraint(epsilon)
        #p.update_bound(D)
        D, P = p.solve(verbose =0)
        print "(D, P) = ", D, ",", P
