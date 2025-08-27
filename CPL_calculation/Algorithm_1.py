'''
    Algorithm 1 
    (Actual CPL using transition probability distribution of LDP mechanism 
    and proabability distribution between attributes).
'''

import numpy as np

def compute_leakage(C, G, G_prime):
    return np.sum(C*G)/np.sum(C*G_prime)

def Algo3_privacy_leakage(Q, CMF, eps):
    
    num_rows, _ = np.shape(CMF)
    _, num_columns_Q = np.shape(Q)
    l = 0
    
    for k in range(num_columns_Q):
        C = Q[:,k]
        for i in range(num_rows):
            G = CMF[i,:]
            for j in range(num_rows):
                if i == j:
                    continue
                G_prime = CMF[j,:]
                
                l = max(l, compute_leakage(C, G, G_prime))

    leakage = min(eps, max(0, np.log(l)))

    return leakage
 