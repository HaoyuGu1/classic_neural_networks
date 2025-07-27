import GP
import random
import numpy as np
import math

def compute_bases(v_list, poly_degree = 30, a = 0.001, b = 20):
    Psi = GP.gp_eigen_funcs_fast(v_list, poly_degree = poly_degree, a = a, b = b)
    lam = GP.gp_eigen_value(poly_degree = poly_degree, a = a, b = b, d = np.array(v_list).shape[1])
    lam = list(lam)
    sqrt_lambda = list(np.sqrt(lam))
    Psi = np.transpose(np.array(Psi))
    Bases = np.zeros((Psi.shape[0], Psi.shape[1]))
    for i in range(len(sqrt_lambda)):
        Bases[i,:] = Psi[i,:] * sqrt_lambda[i]
    return Bases


def simulate_data(n, r2, dim, random_seed = 2023):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    v_list1 = GP.generate_grids(d = 2, num_grids = dim, grids_lim = np.array([-3, 3]))
    v_list2 = GP.generate_grids(d = 2, num_grids = dim, grids_lim = np.array([-1, 1]))
    true_beta1 = (0.5 * v_list1[:, 0] ** 2 + v_list1[:, 1] ** 2) < 2
    true_beta2 = np.exp(-5 * (v_list2[:, 0] - 1.5 * np.sin(math.pi * np.abs(v_list2[:, 1])) + 1.0) **2 )
    p1 = v_list1.shape[0]
    p2 = v_list2.shape[0]

    Bases1 = compute_bases(v_list1)
    Bases2 = compute_bases(v_list2)
    theta1 = np.random.normal(size = n * Bases1.shape[0], scale = 1 / np.sqrt(p1))
    theta1 = theta1.reshape(Bases1.shape[0], n)
    theta2 = np.random.normal(size = n * Bases2.shape[0], scale = 1 / np.sqrt(p2))
    theta2 = theta2.reshape(Bases2.shape[0], n)
    # simulate an image
    img1 = np.transpose(np.dot(np.transpose(Bases1), theta1))
    img2 = np.transpose(np.dot(np.transpose(Bases2), theta2))
    
    # variance of sigma^2
    theta0 =  2
    mean_Y = theta0 + np.dot(img1,true_beta1) + np.dot(img2, true_beta2)
    # mean_Y = theta0 + np.dot(img2,true_beta2)
    true_sigma2 = np.var(mean_Y) * (1 - r2) / r2
    Y = mean_Y + np.random.normal(size = n, scale = np.sqrt(true_sigma2))
    v_list = [v_list1, v_list2]
    true_beta = [true_beta1, true_beta2]
    img = [img1, img2]
    
    return v_list, true_beta, img, Y