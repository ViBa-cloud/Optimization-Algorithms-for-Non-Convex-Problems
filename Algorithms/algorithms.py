# IOE 511/MATH 562, University of Michigan
# Code written by: Chenfei Li, Rachit Garg, Vinayak Bassi
import numpy as np
import time
# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent
# Input: x, f, g
# Output: x_new, f_new, g_new, d, alpha, alg_time, num_f, num_g
def GDStep(x,f,g,problem,method,options):
    # start timing 
    start = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial gradient evluation counter to be 1
    num_f = 1
    num_g = 1
    # Set the search direction d to be -g
    d = -g 

    # determine step sype and step size
    # If the method is "Constant", alpha = options.constant_step_size, the default value of constant step size
    if method.step_type == 'Constant':
        alpha = options.constant_step_size
        x_new = x + alpha*d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

    # If the method is "Backtracking", set the initial alpha to be options.alpha_bar,
    # then iterates until satisfies the Armijo condition
    elif method.step_type == 'Backtracking':
        alpha = options.alpha_bar
        c_1 = options.c_1_ls
        tau = options.tau_ls
        # Whenever go thtough the while loop, alpha becomes alpha * tau
        #                                     number of function evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1 * alpha * np.matmul(np.transpose(g), d)[0][0]:
            alpha = alpha * tau
            num_f = num_f + 1
        # Calculate new x value, f value, and g value
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

    # If the method is "Wolfe", set the initial alpha to be options.alpha_bar,
    # then iterates until satisfies the Wolfe conditions
    elif method.step_type == "Wolfe":
        alpha = options.alpha_bar
        c_1_w = options.c_1_ls
        c_2_w = options.c_2_ls
        c = options.c_ls_w
        alpha_low = options.alpha_low
        alpha_high = options.alpha_high
        # Whenever go thtough the while loop, number of function evaluations adds 1 and number of gradient evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0] or \
              np.matmul(np.transpose(problem.compute_g(x + alpha * d)), d)[0][0] < c_2_w * np.matmul(np.transpose(g), d)[0][0]:
            num_f = num_f + 1
            num_g = num_g + 1
            # If Armijo condition satisfied, evaluates gradient at x + alpha * d, and number of gradient evaluations adds 1
            if problem.compute_f(x + alpha * d)<= f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0]:
                g_temp = problem.compute_g(x + alpha * d)
                num_g = num_g + 1
                if np.matmul(np.transpose(g_temp), d)[0][0] >= c_2_w * np.matmul(np.transpose(g), d)[0][0]:
                    break
                alpha_low = alpha
            else:
                alpha_high = alpha
            # alpha becomes c * alpha_low + (1 - c) * alpha_high
            alpha = c * alpha_low + (1 - c) * alpha_high
        # Calculate new x value, f value, and g value
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)               
    else:
        print('Warning: step type is not defined')
    # stop timing and calculate the algorithm's time
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, d, alpha, alg_time, num_f, num_g

#(2) Newton
# Input: f, g, H, problem, method, options
# Output: x_new, f_new, g_new, H_new, d, alpha, alg_time, num_f, num_g
def Newton(x, f, g, H, problem, method, options):
    # start timing
    start  = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial gradient evluation counter to be 1
    num_f = 1
    num_g = 1
    # Set the search direction d to be -inv(H) * g. 
    d = np.matmul(np.linalg.inv(H), -g)
    
    # If the method is "Backtracking", set the initial alpha to be options.alpha_bar,
    # then iterates until it satisfies the Armijo condition
    if method.step_type == 'Backtracking':
        alpha = options.alpha_bar
        c_1 = options.c_1_ls
        tau = options.tau_ls
        # Whenever go thtough the while loop, alpha becomes alpha * tau
        #                                     number of function evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1 * alpha * np.matmul(np.transpose(g), d)[0][0]:
            alpha = alpha * tau
            num_f = num_f + 1
        # Calculate new x value, f value, and g value
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        H_new = problem.compute_H(x_new)
        
    # If the method is "Wolfe", set the initial alpha to be options.alpha_bar,
    # then iterates until it satisfies the Wolfe conditions
    elif method.step_type == 'Wolfe':
        alpha = options.alpha_bar
        c_1_w = options.c_1_ls
        c_2_w = options.c_2_ls
        c = options.c_ls_w
        alpha_low = options.alpha_low
        alpha_high = options.alpha_high
        # Whenever go thtough the while loop, number of function evaluations adds 1 and number of gradient evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0] or \
              np.matmul(np.transpose(problem.compute_g(x + alpha * d)), d)[0][0] < c_2_w * np.matmul(np.transpose(g), d)[0][0]:
            num_f = num_f + 1
            num_g = num_g + 1
            # If Armijo condition satisfied, evaluates gradient at x + alpha * d, and number of gradient evaluations adds 1
            if problem.compute_f(x + alpha * d)<= f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0]:
                g_temp = problem.compute_g(x + alpha * d)
                num_g = num_g + 1
                if np.matmul(np.transpose(g_temp), d)[0][0] >= c_2_w * np.matmul(np.transpose(g), d)[0][0]:
                    break
                alpha_low = alpha
            else:
                alpha_high = alpha
            # alpha becomes c * alpha_low + (1 - c) * alpha_high
            alpha = c * alpha_low + (1 - c) * alpha_high
        # Calculate new x value, f value, and g value
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        H_new = problem.compute_H(x_new)
    else:
        print('Warning: step type is not defined')
    # stop timing and calculate the algorithm's time
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, H_new, d, alpha, alg_time, num_f, num_g

#(3) ModifiedNewton
# Input: x, f, g, H, k, problem, method, options
# Output: x_new, f_new, g_new, H_new, d, alpha, eta, alg_time, num_f, num_g
def ModifiedNewton(x, f, g, H, k, problem, method, options):
    # start timing
    start = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial gradient evalution counter to be 1
    num_f = 1
    num_g = 1
    # If the method is "Backtracking", set the initial alpha to be options.alpha_bar,
    # then iterates until it satisfies the Armijo condition
    if method.step_type == 'Backtracking':
        beta = options.beta_Newton
        # Find eta, which makes H + eta * I to be positive definite 
        if min(H.diagonal()) > 0:
            eta = 0
        else:
            eta = -min(H.diagonal()) + beta
        
        while min(np.linalg.eigvals(H + eta * np.identity(len(x)))) <= 0:
            eta = max(2 * eta, beta)
            
        # Calculate inv(H + eta * I)
        H_m = np.linalg.inv(H + eta * np.identity(len(x)))
        # Set search direction to be -inv(H + eta * I) * g
        d = np.matmul(H_m, -g)
        alpha = options.alpha_bar
        c_1 = options.c_1_ls
        tau = options.tau_ls
        # Whenever go thtough the while loop, alpha becomes alpha * tau
        #                                     number of function evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1 * alpha * np.matmul(np.transpose(g), d)[0][0]:
            alpha = alpha * tau
            num_f = num_f + 1
        # Calculate new x value, f value, and g value
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        H_new = problem.compute_H(x_new)

    # If the method is "Wolfe", set the initial alpha to be options.alpha_bar,
    # then iterates until satisfies the Wolfe conditions
    elif method.step_type == 'Wolfe':
        beta = options.beta_Newton
        # Find eta, which makes H + eta * I to be positive definite 
        if min(H.diagonal()) > 0:
            eta = 0
        else:
            eta = -min(H.diagonal()) + beta
        
        while min(np.linalg.eigvals(H + eta * np.identity(len(x)))) <= 0:
            eta = max(2 * eta, beta)
            
        # Calculate inv(H + eta * I)
        H_m = np.linalg.inv(H + eta * np.identity(len(x)))
        # Set search direction to be -inv(H + eta * I) * g
        d = np.matmul(H_m, -g)
        alpha = options.alpha_bar
        c_1_w = options.c_1_ls
        c_2_w = options.c_2_ls
        c = options.c_ls_w
        alpha_low = options.alpha_low
        alpha_high = options.alpha_high
        # Whenever go thtough the while loop, number of function evaluations adds 1 and number of gradient evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0] or \
              np.matmul(np.transpose(problem.compute_g(x + alpha * d)), d)[0][0] < c_2_w * np.matmul(np.transpose(g), d)[0][0]:
            num_f = num_f + 1
            num_g = num_g + 1
            # If Armijo condition satisfied, evaluates gradient at x + alpha * d, and number of gradient evaluations adds 1
            if problem.compute_f(x + alpha * d)<= f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0]:
                g_temp = problem.compute_g(x + alpha * d)
                num_g = num_g + 1
                if np.matmul(np.transpose(g_temp), d)[0][0] >= c_2_w * np.matmul(np.transpose(g), d)[0][0]:
                    break
                alpha_low = alpha
            else:
                alpha_high = alpha
            # alpha becomes c * alpha_low + (1 - c) * alpha_high    
            alpha = c * alpha_low + (1 - c) * alpha_high
        # Calculate new x value, f value, and g value
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        H_new = problem.compute_H(x_new)
        
    else:
        print('Warning: step type is not defined')
    # stop timing and calculate the algorithm's time
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, H_new, d, alpha, eta, alg_time, num_f, num_g


#(4) BFGS
# Input: x, f, g, H_list, epsilon, problem, method, options
# Output: x_new, f_new, g_new, H_list, d, alpha, alg_time, num_f, num_g
def BFGS(x, f, g, H_list, epsilon, problem, method, options):
    # H_list is a list of estimations of H, the inverse of Hessian matrix
    # start timing
    start  = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial gradient evalution counter to be 1
    num_f = 1
    num_g = 1
    # Set the search direction d to be -H * g. 
    d = np.matmul(H_list[len(H_list) - 1], -g)
    H = H_list[len(H_list) - 1]
    
    # If the method is "Backtracking", set the initial alpha to be options.alpha_bar,
    # then iterates until it satisfies the Armijo condition
    if method.step_type == 'Backtracking':
        alpha = options.alpha_bar
        c_1 = options.c_1_ls
        tau = options.tau_ls
        # Whenever go thtough the while loop, alpha becomes alpha * tau
        #                                     number of function evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1 * alpha * np.matmul(np.transpose(g), d)[0][0]:
            alpha = alpha * tau
            num_f = num_f + 1
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        # Calculate s_k = x_new - x, y_k = g_new - g
        s_k = x_new - x
        y_k = g_new - g
        I_n = np.identity(len(x))
        # If s_{k}^{T}y_k > epsilon * ||s_k||_{2}^{2} * ||y_{k}||_{2}^{2}, update H. If not, does not update H. 
        if np.matmul(np.transpose(s_k), y_k)[0][0] > epsilon * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            M_1 = I_n - np.matmul(s_k, np.transpose(y_k))/np.matmul(np.transpose(s_k), y_k)[0][0]
            M_2 = I_n - np.matmul(y_k, np.transpose(s_k))/np.matmul(np.transpose(s_k), y_k)[0][0]
            H_new = np.matmul(np.matmul(M_1, H), M_2) + np.matmul(s_k, np.transpose(s_k))/np.matmul(np.transpose(s_k), y_k)[0][0]
            H_list.append(H_new)
            
    # If the method is "Wolfe", set the initial alpha to be options.alpha_bar,
    # then iterates until satisfies the Wolfe conditions     
    elif method.step_type == 'Wolfe':
        alpha = options.alpha_bar
        c_1_w = options.c_1_ls
        c_2_w = options.c_2_ls
        c = options.c_ls_w
        alpha_low = options.alpha_low
        alpha_high = options.alpha_high
        # Whenever go thtough the while loop, number of function evaluations adds 1 and number of gradient evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0] or \
              np.matmul(np.transpose(problem.compute_g(x + alpha * d)), d)[0][0] < c_2_w * np.matmul(np.transpose(g), d)[0][0]:
            num_f = num_f + 1
            num_g = num_g + 1
            # If Armijo condition satisfied, evaluates gradient at x + alpha * d, and number of gradiene evaluations adds 1
            if problem.compute_f(x + alpha * d)<= f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0]:
                g_temp = problem.compute_g(x + alpha * d)
                num_g = num_g + 1
                if np.matmul(np.transpose(g_temp), d)[0][0] >= c_2_w * np.matmul(np.transpose(g), d)[0][0]:
                    break
                alpha_low = alpha
            else:
                alpha_high = alpha
            # alpha becomes c * alpha_low + (1 - c) * alpha_high
            alpha = c * alpha_low + (1 - c) * alpha_high
        # Calculate new x value, f value, and g value
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        # Calculate s_k = x_new - x, y_k = g_new - g
        s_k = x_new - x
        y_k = g_new - g
        I_n = np.identity(len(x))
        # If s_{k}^{T}y_k > epsilon * ||s_k||_{2}^{2} * ||y_{k}||_{2}^{2}, update H using the BFGS algorithm, and add the new H as the last element of the H list. If not, does not update H. 
        if np.matmul(np.transpose(s_k), y_k)[0][0] > epsilon * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            M_1 = I_n - np.matmul(s_k, np.transpose(y_k))/np.matmul(np.transpose(s_k), y_k)[0][0]
            M_2 = I_n - np.matmul(y_k, np.transpose(s_k))/np.matmul(np.transpose(s_k), y_k)[0][0]
            H_new = np.matmul(np.matmul(M_1, H), M_2) + np.matmul(s_k, np.transpose(s_k))/np.matmul(np.transpose(s_k), y_k)[0][0]
            H_list.append(H_new)

         
    else:
        print('Warning: step type is not defined')
    # stop timimg and calculate the algorithm's time
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, H_list, d, alpha, alg_time, num_f, num_g

#(5) DFP
# Input: x, f, g, H_list, epsilon, problem, method, options
# Output: x_new, f_new, g_new, H_list, d, alpha, alg_time, num_f, num_g
def DFP(x, f, g, H_list, epsilon, problem, method, options):
    # H_list is a list of estimations of H, the inverse of Hessian matrix
    # start timing 
    start = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial gradient evalution counter to be 1
    num_f = 1
    num_g = 1
    # Set the search direction d to be -H * g. 
    d = np.matmul(H_list[len(H_list) - 1], -g)
    H = H_list[len(H_list) - 1]

    # If the method is "Backtracking", set the initial alpha to be options.alpha_bar,
    # then iterates until it satisfies the Armijo condition
    if method.step_type == 'Backtracking':
        alpha = options.alpha_bar
        c_1 = options.c_1_ls
        tau = options.tau_ls
        # Whenever go thtough the while loop, alpha becomes alpha * tau
        #                                     number of function evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1 * alpha * np.matmul(np.transpose(g), d)[0][0]:
            alpha = alpha * tau
            num_f = num_f + 1
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        # Calculate s_k = x_new - x, y_k = g_new - g
        s_k = x_new - x
        y_k = g_new - g
        # If s_{k}^{T}y_k > epsilon * ||s_k||_{2}^{2} * ||y_k||_{2}^{2}, update H, and adds the new H as the last element of the H list. If not, does not update H.
        if np.matmul(np.transpose(s_k), y_k)[0][0] > epsilon * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            H_new = H - (np.matmul(np.matmul(np.matmul(H, y_k), np.transpose(y_k)), H)/np.matmul(np.matmul(np.transpose(y_k), H), y_k)[0][0]) + (np.matmul(s_k, np.transpose(s_k))/np.matmul(np.transpose(s_k), y_k)[0][0])
            H_list.append(H_new)

    # If the method is "Wolfe", set the initial alpha to be options.alpha_bar,
    # then iterates until satisfies the Wolfe conditions     
    elif method.step_type == 'Wolfe':
        alpha = options.alpha_bar
        c_1_w = options.c_1_ls
        c_2_w = options.c_2_ls
        c = options.c_ls_w
        alpha_low = options.alpha_low
        alpha_high = options.alpha_high
        # Whenever go thtough the while loop, number of function evaluations adds 1 and number of gradient evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0] or \
              np.matmul(np.transpose(problem.compute_g(x + alpha * d)), d)[0][0] < c_2_w * np.matmul(np.transpose(g), d)[0][0]:
            num_f = num_f + 1
            num_g = num_g + 1
            # If Armijo condition satisfied, evaluates gradient at x + alpha * d, and number of gradiene evaluations adds 1
            if problem.compute_f(x + alpha * d)<= f + c_1_w * alpha * np.matmul(np.transpose(problem.compute_g(x)), d)[0][0]:
                g_temp = problem.compute_g(x + alpha * d)
                num_g = num_g + 1
                if np.matmul(np.transpose(g_temp), d)[0][0] >= c_2_w * np.matmul(np.transpose(g), d)[0][0]:
                    break
                alpha_low = alpha
            else:
                alpha_high = alpha
            # alpha becomes c * alpha_low + (1 - c) * alpha_high
            alpha = c * alpha_low + (1 - c) * alpha_high
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        # Calculate s_k = x_new - x, y_k = g_new - g
        s_k = x_new - x
        y_k = g_new - g
        # If s_{k}^{T}y_k > epsilon * ||s_k||_{2} * ||y_k||_{2}, update H using the DFP algorithm, and adds the new H as the last element of the H list. If not, does not update H.
        if np.matmul(np.transpose(s_k), y_k)[0][0] > epsilon * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            H_new = H - (np.matmul(np.matmul(np.matmul(H, y_k), np.transpose(y_k)), H)/np.matmul(np.matmul(np.transpose(y_k), H), y_k)[0][0]) + (np.matmul(s_k, np.transpose(s_k))/np.matmul(np.transpose(s_k), y_k)[0][0])
            H_list.append(H_new)
         
    else:
        print('Warning: step type is not defined')
    # stop timing and calculate the algorithm's time
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, H_list, d, alpha, alg_time, num_f, num_g

# (6) L_BFGS
# Input: x, f, g, s, y, k,epsilon, problem, method, options
# Output: x_new, f_new, g_new, s, y, d, k, alpha, alg_time, num_f, num_g
def L_BFGS(x, f, g, s, y, k,epsilon, problem, method, options):
    # start timing 
    start = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial gradient evalution counter to be 1
    num_f = 1
    num_g = 1
    # s is a list containing the m most recent s_k
    # y is a list containing the m most recent y_k
    #print(x)
    q = g
    m = options.m_L_BFGS
    # Set the initial search direction d to be -H0 * g. 
    a = []
    if k == 0:
        d = np.identity(len(x)) @ (-g)
    else:
        # Two for loops to find r = H_k * g
        for i in range (len(s) - 1, -1, -1):
            a_i = (np.transpose(s[i]) @ q)[0][0] /(np.transpose(s[i]) @ y[i])[0][0]
            a.insert(0, a_i)
            q = q - a_i * y[i]
        r = np.identity(len(x)) @ q

        for i in range (0, len(s)):
            beta = (np.transpose(y[i])@ r)[0][0]/(np.transpose(s[i]) @ y[i])[0][0]
            r = r + s[i] * (a[i] - beta)
        # Set the search direction be d = -r
        d = -r
        
    # If the method is "Backtracking", set the initial alpha to be options.alpha_bar,
    # then iterates until it satisfies the Armijo condition
    if method.step_type == 'Backtracking':
        alpha = options.alpha_bar
        c_1 = options.c_1_ls
        tau = options.tau_ls
        # Whenever go thtough the while loop, alpha becomes alpha * tau
        #                                     number of function evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1 * alpha * np.matmul(np.transpose(g), d)[0][0]:
            alpha = alpha * tau
            num_f = num_f + 1

        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        s_k = x_new - x
        y_k = g_new - g
        # If s_{k}^{T}y_k > epsilon * ||s_k||_{2} * ||y_{k}||_{2}, add s_k and y_k to the s list and y list. If not, does not add s_k to s or add y_k to y.
        if np.matmul(np.transpose(s_k), y_k)[0][0] > epsilon * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            s.append(s_k)
            y.append(y_k)
        # Set the s to be the list of m most recent s_k, and set y to be the list of m most recent y_k. 
        s = s[max(0, len(s) - m):len(s)]
        y = y[max(0, len(y) - m):len(y)]

    # If the method is "Wolfe", set the initial alpha to be options.alpha_bar,
    # then iterates until satisfies the Wolfe conditions
    elif method.step_type == 'Wolfe':
        alpha = options.alpha_bar
        c_1_w = options.c_1_ls
        c_2_w = options.c_2_ls
        c = options.c_ls_w
        alpha_low = options.alpha_low
        alpha_high = options.alpha_high
        # Whenever go thtough the while loop, number of function evaluations adds 1 and number of gradient evaluations adds 1
        while problem.compute_f(x + alpha * d) > f + c_1_w * alpha * np.matmul(np.transpose(g), d)[0][0] or \
              np.matmul(np.transpose(problem.compute_g(x + alpha * d)), d)[0][0] < c_2_w * np.matmul(np.transpose(g), d)[0][0]:
            num_f = num_f + 1
            num_g = num_g + 1
            # If Armijo condition satisfied, evaluates gradient at x + alpha * d, and number of gradiene evaluations adds 1
            if problem.compute_f(x + alpha * d)<= f + c_1_w * alpha * np.matmul(np.transpose(problem.compute_g(x)), d)[0][0]:
                g_temp = problem.compute_g(x + alpha * d)
                num_g = num_g + 1
                if np.matmul(np.transpose(g_temp), d)[0][0] >= c_2_w * np.matmul(np.transpose(g), d)[0][0]:
                    break
                alpha_low = alpha
            else:
                alpha_high = alpha
            # alpha becomes c * alpha_low + (1 - c) * alpha_high
            alpha = c * alpha_low + (1 - c) * alpha_high
        x_new = x + alpha * d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        # Calculate s_k = x_new - x, y_k = g_new - g
        s_k = x_new - x
        y_k = g_new - g
        # If s_{k}^{T}y_k > epsilon * ||s_k||_{2} * ||y_{k}||_{2}, add s_k and y_k to the s list and y list. If not, does not add s_k and y_k to the s list and y list.
        if np.matmul(np.transpose(s_k), y_k)[0][0] > epsilon * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            s.append(s_k)
            y.append(y_k)
        # Set the s to be the list of m most recent s_k, and set y to be the list of m most recent y_k. 
        s = s[max(0, len(s) - m):len(s)]
        y = y[max(0, len(y) - m):len(y)]
    else:
        print('Warning: step type is not defined')
    # stop timing and calculate the algorithm's time
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, s, y, d, k, alpha, alg_time, num_f, num_g

# (7) TRNewtonCG
# Input: x, f, g, H, delta, term_tol_CG, problem, method, options
# Output: x_new, f_new, g_new, H_new, delta, rho_k, alg_time, num_f, num_g
def TRNewtonCG(x, f, g, H, delta, term_tol_CG, problem, method, options):
    # start timing
    start  = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial function evalution counter to be 1
    num_f = 1
    num_g = 1
    # solve CG subproblem and find d
    d = CG(x, f, g, H, delta, term_tol_CG, options)
    # calculate m and rho_k
    m = f + np.matmul(np.transpose(g), d)[0][0] + 1/2 * (np.transpose(d) @ H @ d)[0][0]
    rho_k = (f - problem.compute_f(x + d))/(f - m)
    c1_CG = options.c_1_tr
    c2_CG = options.c_2_tr
    # If rho_k is larger than c1, update x_new = x + d, and if rho_k is larger than c2, set delta, the radius of trust region, to be 2 * delta.
    # If rho_k is smaller than c1, do not update x_new, and set delta = 1/2 * delta. 
    if rho_k > c1_CG:
        x_new = x + d
        if rho_k > c2_CG:
            delta = 2 * delta
    else:
        x_new = x
        delta = 1/2 * delta
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    H_new = problem.compute_H(x_new)
    # stop timing and calculate the algorithm's time
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, H_new, delta, rho_k, alg_time, num_f, num_g

# (8) TRSR1CG
# Input: x, f, g, B_list, delta, term_tol_CG, problem, method, options
# Output: x_new, f_new, g_new, B_list, delta, rho_k, alg_time, num_f, num_g
def TRSR1CG(x, f, g, B_list, delta, term_tol_CG, problem, method, options):
    # start timing 
    start  = time.time()
    # num_f is the number of function evaluations, at least we need to calculate f(x_new), so set the initial function evalution counter to be 1
    # num_g is the number of gradient evaluations, at least we need to calculate g(x_new), so set the initial function evalution counter to be 1
    num_f = 1
    num_g = 1
    # B is a list of B, the estimation of Hessian. 
    B = B_list[len(B_list) - 1]
    # solve CG subproblem to find d
    d = CG(x, f, g, B, delta, term_tol_CG, options)
    # calculate m and rho_k
    m = f + np.matmul(np.transpose(g), d)[0][0] + 1/2 * (np.transpose(d) @ B @ d)[0][0]
    rho_k = (f - problem.compute_f(x + d))/(f - m)
    c1_CG = options.c_1_tr
    c2_CG = options.c_2_tr
    # If rho_k is larger than c1, update x_new = x + d, and if rho_k is larger than c2, set delta, the radius of trust region, to be 2 * delta.
    # If rho_k is smaller than c1, do not update x_new, and set delta = 1/2 * delta.
    if rho_k > c1_CG:
        x_new = x + d
        if rho_k > c2_CG:
            delta = 2 * delta
    else:
        x_new = x
        delta = 1/2 * delta
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    # Calculate y = g_new - g, s = x_new - x
    y = g_new - g
    s = x_new - x
    c3_CG = options.c_3_tr
    # If |(y - B * s) * s| > c_3 * ||y - B * s||_{2} * ||s||_{2}, update B using SR1 method. If not, does not update B. 
    if abs((np.transpose(y - B @ s) @ s)[0][0]) <= c3_CG * np.linalg.norm(y - B @ s) * np.linalg.norm(s):
        B_new = B
    else:
        B_new = B + (y - B @ s) * np.transpose((y - B @ s))/(np.transpose(y - B @ s) @ s)[0][0]
        B_list.append(B_new)
    # stop timing and calculate the algorithm's time 
    stop = time.time()
    alg_time = stop - start
    return x_new, f_new, g_new, B_list, delta, rho_k, alg_time, num_f, num_g
    
# CG function
# Input: x, f, g, B, delta, term_tol_CG, options
# Output: d
def CG(x, f, g, B, delta, term_tol_CG, options):
    # set initial z to be a zero vector, set initial r = g, and set initial rho = -g
    n = len(x)
    z = np.zeros((n, 1))
    r = g
    rho = -g
    # If norm of r is less than term_tol_CG, set the search direction to be 0
    if np.linalg.norm(r) < term_tol_CG:
        d = np.zeros((n, 1))
        return d
    else:
        # i is the number of CG iterations
        i = 1
        # While rho * B * rho > 0 and the number of iterations is less than max_iterations_CG
        while (np.transpose(rho) @ B @ rho)[0][0] > 0 and i < options.max_iterations_CG:
            # Calculate alpha and z_new
            alpha = (np.transpose(r) @ r)[0][0]/(np.transpose(rho) @ B @ rho)[0][0]
            z_new = z + alpha * rho
            # If ||z_new||_{2} >= delta, find tau1, tau2, the two values of tau that makes d = z + delta * rho satisfied ||d||_{2} = delta.
            # Choose the tau that gives smaller value of m_k(d) = f(x_k) + \nabla f(x_k)^{T}d + 1/2 * d^{T} * B_k * d.
            # return d = z + tau * rho. 
            if np.linalg.norm(z_new) >= delta:
                z_squared = (np.transpose(z) @ z)[0][0] 
                rho_squared = (np.transpose(rho) @ rho)[0][0] 
                squared = (np.transpose(z) @ rho)[0][0] ** 2 - z_squared * rho_squared + rho_squared * delta ** 2
                tau1 = (-(np.transpose(z) @ rho)[0][0] + np.sqrt(squared))/(np.transpose(rho) @ rho)[0][0]
                d1 = z + tau1 * rho
                tau2 = (-(np.transpose(z) @ rho)[0][0] - np.sqrt(squared))/(np.transpose(rho) @ rho)[0][0]
                d2 = z + tau2 * rho
                if (f + (np.transpose(g) @ d1)[0][0] + (1/2 * np.transpose(d1) @ B @ d1)[0][0]) \
                   <= (f + (np.transpose(g) @ d2)[0][0] + (1/2 * np.transpose(d2) @ B @ d2)[0][0]):
                    tau = tau1
                else:
                    tau = tau2
                d = z + tau * rho
                return d
            # Set r_new = r + alpha * B * rho
            r_new = r + alpha * B @ rho
            # If ||r_new||_{2} <= term_tol_CG, return d = z_new. 
            if np.linalg.norm(r_new) <= term_tol_CG:
                d = z_new
                return d
            # update rho and r
            beta = (np.transpose(r_new)@r_new)[0][0]/(np.transpose(r) @ r)[0][0]
            rho = -(r + alpha * B @ rho) + beta * rho
            r = r_new
            z = z_new
            i = i + 1
        # If rho * B * rho <= 0 or the number of iterations is larger than max_iterations_CG, find tau1, tau2, the two values of tau that makes d = z + delta * rho satisfied ||d||_{2} = delta.
        # Choose the tau that gives smaller value of m_k(d) = f(x_k) + \nabla f(x_k)^{T}d + 1/2 * d^{T} * B_k * d.
        # return d = z + tau * rho. 
        z_squared = (np.transpose(z) @ z)[0][0] 
        rho_squared = (np.transpose(rho) @ rho)[0][0] 
        squared = (np.transpose(z) @ rho)[0][0] ** 2 - z_squared * rho_squared + rho_squared * delta ** 2
        tau1 = (-(np.transpose(z) @ rho)[0][0] + np.sqrt(squared))/(np.transpose(rho) @ rho)[0][0]
        d1 = z + tau1 * rho
        tau2 = (-(np.transpose(z) @ rho)[0][0] - np.sqrt(squared))/(np.transpose(rho) @ rho)[0][0]
        d2 = z + tau2 * rho
        if (f + (np.transpose(g) @ d1)[0][0] + (1/2 * np.transpose(d1) @ B @ d1)[0][0]) <= (f + (np.transpose(g) @ d2)[0][0] + (1/2 * np.transpose(d2) @ B @ d2)[0][0]):
            tau = tau1
        else:
            tau = tau2
        d = z + tau * rho
        return d
