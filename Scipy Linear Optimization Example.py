from pulp import *
import glpk 
import pandas as pd
import numpy as np
import scipy as sci
from scipy.optimize import linprog

import matplotlib.pyplot as plt
import seaborn as sns


obj = [-1, -2] # objective function

lhs_ineq = [[ 2,  1],  # Red constraint left side
            [-4,  5],  # Blue constraint left side
           [ 1, -2]]  # Yellow constraint left side
           
rhs_ineq = [20,  # Red constraint right side
            10,  # Blue constraint on right side
            2]  # Yellow constraint right side
            
lhs_eq = [[-1, 5]]

rhs_eq = [15]


bnd = [(0, float("inf")),  # Bounds of x
        (0, float("inf"))]  # Bounds of y
        
        
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
               A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
               method="highs")

opt

`````````````````````````
output:
`````````````````````````
`````````````````````````

        message: Optimization terminated successfully. (HiGHS Status 7: Optimal)
        success: True
         status: 0
            fun: -16.818181818181817
              x: [ 7.727e+00  4.545e+00]
            nit: 0
          lower:  residual: [ 7.727e+00  4.545e+00]
                 marginals: [ 0.000e+00  0.000e+00]
          upper:  residual: [       inf        inf]
                 marginals: [ 0.000e+00  0.000e+00]
          eqlin:  residual: [ 0.000e+00]
                 marginals: [-2.727e-01]
        ineqlin:  residual: [ 0.000e+00  1.818e+01  3.364e+00]
                 marginals: [-6.364e-01 -0.000e+00 -0.000e+00]
 mip_node_count: 0
 mip_dual_bound: 0.0
        mip_gap: 0.0