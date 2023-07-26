# IOE 511/MATH 562, University of Michigan
# Code written by: Chenfei Li, Rachit Garg, Vinayak Bassi

import numpy as np
import project_problems
from algorithms import GDStep, Newton, ModifiedNewton, BFGS, L_BFGS, DFP, TRNewtonCG, TRSR1CG

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)
def optSolver_Dynamix(problem,method,options):
    x_list = []
    f_value_list = []
    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    # If Newton's methods, compute Hessian
    if method.name == 'Newton' or method.name == "ModifiedNewton" or method.name == "TRNewtonCG":
        H = problem.compute_H(x)
    
    norm_g = np.linalg.norm(g, ord = np.inf)
    norm_g_0 = np.linalg.norm(g, ord = np.inf)

    # x_list is the list of x, and f_list is the list of f values
    x_list.append(x)
    f_value_list.append(f)
    # H_list is the list of H, estimation of hessian inverse, and B_list is the list of B, estimation of hessian
    # We set H0 and B0 to be identity matrix 
    H_list = [np.identity(len(x))]
    B_list = [np.identity(len(x))]
    # set initial delta, the radius of trust region;
    #             epsilon, the tolerance of update inverse hessian estimation H for BFGS and DFP algorithms
    #             s is the list of s_k = x_{k + 1} - x_k, and y is the list of y_k = \nabla f(x_{k + 1}) - \nabla f(x_k)
    delta = options.delta_tr
    epsilon = options.term_tol
    tem_tol_CG = options.term_tol_CG
    s = []
    y = []
    # set initial iteration counter, initial time counter, initial function evaluation counter, and initial gradient evaluation counter
    # Because we already evaluted f and g once, we set the initial function evaluation counter, and initial gradient evaluation counter to be 1 
    k = 0
    time_all = 0
    num_f_all = 1
    num_g_all = 1
    # The stop criteria is the number of iterations larger than max_iterations or norm of g is smaller than or equal to term_tol * max(1, norm_g_0)
    # The allowded method names are 'GradientDescent', 'Newton' 'ModifiedNewton', 'L_BFGS', 'TRNewtonCG', 'TRSR1CG', 'BFGS', 'DFP', and 'L_BFGS'
    while k < options.max_iterations and norm_g > options.term_tol * max(1, norm_g_0):
        if method.name == 'GradientDescent':
            x_new, f_new, g_new, d, alpha, time, num_f, num_g = GDStep(x,f,g,problem,method,options)
        # If Newton's methods, we update hessian
        elif method.name == 'Newton': 
            x_new, f_new, g_new, H_new, d, alpha, time, num_f, num_g = Newton(x,f,g,H, problem,method,options)
            H_old = H
            H = H_new
        elif method.name == 'ModifiedNewton': 
            x_new, f_new, g_new, H_new, d, alpha, eta, time, num_f, num_g = ModifiedNewton(x,f,g,H, k, problem,method,options)
            H_old = H
            H = H_new
        elif method.name == "TRNewtonCG":
            x_new, f_new, g_new, H_new, delta, rho, time, num_f, num_g = TRNewtonCG(x, f, g, H, delta, tem_tol_CG, problem, method, options)
            H_old = H
            H = H_new
        elif method.name == "TRSR1CG":
            x_new, f_new, g_new, B_list, delta, rho, time, num_f, num_g = TRSR1CG(x, f, g, B_list, delta, tem_tol_CG, problem, method, options)
        elif method.name == "BFGS":
            x_new, f_new, g_new, H_list, d, alpha, time, num_f, num_g = BFGS(x, f, g, H_list, epsilon, problem, method, options)
        elif method.name == "DFP":
            x_new, f_new, g_new, H_list, d, alpha, time, num_f, num_g = DFP(x, f, g, H_list, epsilon, problem, method, options)
        elif method.name == "L_BFGS":
            x_new, f_new, g_new, s, y, d, k, alpha, time, num_f, num_g = L_BFGS(x, f, g, s, y, k, epsilon, problem, method, options)
        else:
            print('Warning: method is not implemented yet')
    
        # update old and new function values, old and new gradient, old and new gradient norms for all methods         
        x_old = x; f_old = f; g_old = g; norm_g_old = norm_g; 
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g,ord= np.inf);
        # add new x to x_list, and add new f to f_value_list
        x_list.append(x)
        f_value_list.append(f)

        # increment iteration counter, time counter, function evaluation counter, and gradient evaluation counter
        k = k + 1
        time_all = time_all + time
        num_f_all = num_f_all + num_f
        num_g_all = num_g_all + num_g
    return x,f
