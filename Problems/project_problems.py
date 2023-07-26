# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi (template)
#                  Chenfei Li, Rachit Garg, Vinayak Bassi

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.io
# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1)(2)(3)(4) Quadractic function
# (5)(6) Quartic function
# (7)(8) Rosenbrock function 
# (9) Data fit
# (10)(11) Exponential
# (12) Genhumps_5
 
# Problem Number: 1
# Problem Name: quad_10_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 10

# function that computes the function value of the quad_10_10 function

def quad_10_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))
    k = np.linalg.inv(Q) @ q

    return (1/2*x.T@Q@x + q.T@x)[0][0]

# function that computes the gradient of the quad_10_10 function

def quad_10_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))
    
    return Q@x + q   

# function that computes the hessian of the quad_10_10 function    

def quad_10_10_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))

    return Q

# Problem Number: 2
# Problem Name: quad_10_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_1000 function

 

def quad_10_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0][0]

# function that computes the gradient of the quad_10_1000 function

def quad_10_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))
    
    return Q@x + q

# function that computes the hessian of the quad_10_1000 function

def quad_10_1000_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))

    return Q
# Problem Number: 3
# Problem Name: quad_1000_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 10

# function that computes the function value of the quad_1000_10 function

def quad_1000_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))

    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = mat['Q']
    Q = Q * np.identity(len(x))

    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0][0]

# function that computes the gradient of the quad_1000_10 function

def quad_1000_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))
    
    return Q@x + q

# function that computes the hessian of the quad_1000_10 function

def quad_1000_10_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))

    return Q
# Problem Number: 4
# Problem Name: quad_1000_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_1000_1000 function

def quad_1000_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = mat['Q']
    Q = Q * np.identity(len(x))
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0][0]

# function that computes the gradient of the quad_1000_1000 function

def quad_1000_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))

    return Q@x + q

# function that computes the hessian of the quad_1000_1000 function

def quad_1000_1000_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = mat['Q']

    Q = Q * np.identity(len(x))

    return Q

# Problem Number: 5
# Problem Name: quartic_1
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_1 function


def quartic_1_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    
    return (1/2*(x.T @x) + sigma/4*(x.T@Q@x)**2)[0][0]

# function that computes the gradient of the quartic_1 function

def quartic_1_grad(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    
    return x + sigma *(x.T@Q@x)* Q@x

# function that computes the hessian of the quartic_1 function

def quartic_1_Hess(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    
    return np.identity(4) + 2 * sigma * np.transpose(Q@x) @ (Q@x) + sigma *(x.T@Q@x)* Q

# Problem Number: 6
# Problem Name: quartic_2
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_2 function


def quartic_2_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    
    return (1/2*(x.T@x) + sigma/4*(x.T@Q@x)**2)[0][0]

# function that computes the gradient of the quartic_2 function

def quartic_2_grad(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    
    return x + sigma *(x.T@Q@x)* Q@x

# function that computes the hessian of the quartic_2 function

def quartic_2_Hess(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    
    return np.identity(4) + 2 * sigma * np.transpose(Q@x) @ (Q@x) + sigma *(x.T@Q@x)* Q

# Problem Number: 7 and 8
# Problem Name: Rosenbrock_2 and Rosenbrock_100
# Problem Description: A rosenbrock function. Dimension n = 2 or 100

# function that computes the function value of the Rosenbrock function

def rosen_func(x):
    f = 0
    for i in range(0, len(x) - 1):
        f = f + (1-x[i][0])**2 + 100*(x[i + 1][0] - x[i][0]**2)**2
    return f

# Function that computes the gradient of the Rosenbrock function

def rosen_grad(x):
    n = len(x)
    g = np.zeros((n, 1))
    if n == 2:
        g[0][0] = -2 + 2 * x[0][0] - 400 * x[0][0] * x[1][0] + 400 * x[0][0]**3
        g[n - 1][0] = 200 * x[n - 1][0] - 200 * x[n - 2][0] ** 2
    else:
        g[0][0] = -2 + 2 * x[0][0] - 400 * x[0][0] * x[1][0] + 400 * x[0][0]**3
        g[n - 1][0] = 200 * x[n - 1][0] - 200 * x[n - 2][0] ** 2
        for i in range(1, n - 1):
            g[i][0] = 200 * (x[i][0] - x[i - 1][0] ** 2) + 2 * (x[i][0] - 1) - 400 * (x[i + 1][0] - x[i][0] ** 2) * x[i][0]
    return g

# function that computes the hessian of the Rosenbrock function

def rosen_Hess(x):
    n = len(x)
    H = np.zeros((n, n))
    if n == 2:
        H[0][0] = 2 - 400 * x[1][0] + 1200 * x[0][0]**2
        H[0][1] = -400 * x[0][0]
        H[1][0] = -400 * x[0][0]
        H[1][1] = 200
    else:
        H[0][0] = 2 - 400 * x[1][0] + 1200 * x[0][0]**2
        H[0][1] = -400 * x[0][0]
        H[n - 1][n - 2] = -400 * x[n - 2]
        H[n - 1][n - 1] = 200
        for i in range(1, n - 1):
            H[i][i - 1] = -400 * x[i - 1]
            H[i][i] = 200 + 2 - 400 * x[i + 1][0] + 1200 * x[i][0] ** 2
            H[i][i + 1] = -400 * x[i][0]
    return H


# Problem Number: 9
# Problem Name: data_fit_2
# Problem Description: A quartic function. Dimension n = 3

# function that computes the gradient of function data_fit_2

def data_fit_2_func(x):
    w = x[0][0]
    z = x[1][0]
    return (1.5 - w * (1 - z)) ** 2 + (2.25 - w * (1 - z * z)) ** 2 + (2.625 - w * (1 - z * z * z)) ** 2


# function that computes the gradient of function data_fit_2

def data_fit_2_grad(x):
    w = x[0][0]
    z = x[1][0]   
    g0 = (z - 1) * 2 * (1.5 - w * (1 - z)) + (z * z - 1) * 2 * (2.25 - w * (1 - z * z)) \
         + (z * z * z - 1) * 2 * (2.625 - w * (1 - z * z * z))
    g1 = 2 * w * (1.5 - w * (1 - z)) + 4 * z * w * (2.25 - w * (1 - z * z)) + 6 * z * z * w * (2.625 - w * (1 - z * z * z))
    return np.array([[g0], [g1]])


# function that computes the hessian of function data_fit_2

def data_fit_2_Hess(x):
    w = x[0][0]
    z = x[1][0]   
    h00 = 2 * (z - 1) ** 2 + 2 * (z * z - 1) ** 2 + 2 * (z ** 3 - 1) ** 2 
    h01 = (2 * 1.5  - 4 * w + 4 * z * w) + (4 * z * 2.25 - 8 * z * w + 8 * w * z ** 3) + (6 * 2.625 * z ** 2 - 12 * w * z ** 2 + 12 * w * z ** 5)
    h10 = h01
    h11 = 4 * w + 4 * w * 2.25 - 4 * w * w + 12 * w * w * z * z + 12 * w * 2.625 * z - 12 * w * w * z + 30 * w * w * z ** 4
    return np.array([[h00, h01], [h10, h11]])

# Problem Number: 10 and 11
# Problem Name: Exponential_10 and Exponential_1000
# Problem Description: An exponential function. Dimension n = 2 or 1000

# function that computes the function value of the exponential function. 

def exponential_func(x):
    n = len(x)
    f = 1 - 2/(np.exp(x[0][0]) + 1) + 0.1 * np.exp(-x[0][0])
    for i in range(1, n):
        f = f + (x[i][0] - 1) ** 4
    return f

# function that computes the gradient of exponential function

def exponential_grad(x):
    n = len(x)
    v = np.zeros((n, 1))
    v[0][0] = 2 * np.exp(x[0][0])/((np.exp(x[0][0]) + 1) ** 2) - 0.1 * np.exp(-x[0][0])
    for i in range(1, n):
        v[i][0] = 4 * (x[i][0] - 1) ** 3
    return v

# function that computes the hessian of exponential function

def exponential_Hess(x):
    n = len(x)
    diag = [(2 * np.exp(x[0][0]) - 2 * np.exp(2 * x[0][0]))/((np.exp(x[0][0]) + 1)**3) + 0.1 * np.exp(-x[0][0])]
    for i in range(1, n):
        diag.append(12 * (x[i][0] - 1) ** 2)
    return np.diag(diag)




# Problem Number: 12
# Problem Name: genhumps_5
# Problem Description: A multi-dimensional problem with a lot of humps.
#                      This problem is from the well-known CUTEr test set.

# function that computes the function value of the genhumps_5 function



def genhumps_5_func(x):
    f = 0
    for i in range(4):
        f = f + np.sin(2*x[i])**2*np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
    return f[0]

# function that computes the gradient of the genhumps_5 function
def genhumps_5_grad(x):
    g = [4*np.sin(2*x[0])*np.cos(2*x[0])* np.sin(2*x[1])**2                  + 0.1*x[0],
         4*np.sin(2*x[1])*np.cos(2*x[1])*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2) + 0.2*x[1],
         4*np.sin(2*x[2])*np.cos(2*x[2])*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2) + 0.2*x[2],
         4*np.sin(2*x[3])*np.cos(2*x[3])*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2) + 0.2*x[3],
         4*np.sin(2*x[4])*np.cos(2*x[4])* np.sin(2*x[3])**2                  + 0.1*x[4]]
    
    return np.array(g)

# function that computes the Hessian of the genhumps_5 function
def genhumps_5_Hess(x):
    H = np.zeros((5,5))
    H[0,0] =  8* np.sin(2*x[1])**2*(np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
    H[0,1] = 16* np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])*np.cos(2*x[1])
    H[1,1] =  8*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2)*(np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
    H[1,2] = 16* np.sin(2*x[1])*np.cos(2*x[1])*np.sin(2*x[2])*np.cos(2*x[2])
    H[2,2] =  8*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2)*(np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
    H[2,3] = 16* np.sin(2*x[2])*np.cos(2*x[2])*np.sin(2*x[3])*np.cos(2*x[3])
    H[3,3] =  8*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2)*(np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
    H[3,4] = 16* np.sin(2*x[3])*np.cos(2*x[3])*np.sin(2*x[4])*np.cos(2*x[4])
    H[4,4] =  8* np.sin(2*x[3])**2*(np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
    H[1,0] = H[0,1]
    H[2,1] = H[1,2]
    H[3,2] = H[2,3]
    H[4,3] = H[3,4]
    return H
