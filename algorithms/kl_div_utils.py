from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt

def DiscreteJSD(P, Q):
    M = (P + Q)/2
    return (entropy(P, M) + entropy(Q, M))/2

def GaussKL(mu_p, mu_q, sgm_p, sgm_q):    
    return (np.log(np.linalg.det(sgm_q)) - np.log(np.linalg.det(sgm_p)) - mu_p.shape[0]
            + np.trace(np.linalg.inv(sgm_q) @ sgm_p) +
            (mu_q - mu_p).reshape(1, mu_p.shape[0]) @ np.linalg.inv(sgm_q) @ 
            (mu_q - mu_p).reshape(mu_p.shape[0], 1)
           )/2

def GaussSymmetricKL(P, Q):
    mu_p, sgm_p = P
    mu_q, sgm_q = Q
    return (GaussKL(mu_p, mu_q, sgm_p, sgm_q) + GaussKL(mu_q, mu_p, sgm_q, sgm_p))/2