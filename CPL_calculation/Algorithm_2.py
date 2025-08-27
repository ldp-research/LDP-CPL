'''
    Algorithm 2 - 'def algo2(eps, CMF, delta):'
    (Upper bound for CPL using epsilon, delta and probability distribution between attributes)

    Parameters
        CMF - conditional probability distribution between attributes.
        eps - privacy budget.
        delta - relaxation parameter.

    Return
        (l, f_max) - CPL
'''

import numpy as np

def compute_H(G, G_prime, eps, delta):
    Q = G/(G_prime + 0.000000000000001) # Avoid zero division
    
    sorted_indices = np.argsort(Q)[::-1] # Sorted indexes of Q in decending order

    A = 0
    B = 0
    
    for i in sorted_indices:
        if Q[i] > (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1)):
            A += G[i]
            B += G_prime[i]
        else:
            break

    return (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1)), delta*A

def algo2(eps, CMF, delta):
    num_rows, _ = np.shape(CMF)
    l = 0

    if num_rows == 1:
        return eps, 0, delta
    
    for i in range(num_rows):
        G = CMF[i,:]
        for j in range(num_rows):
            if i == j:
                continue
            G_prime = CMF[j,:]
            H_max, f_ = compute_H(G=G, G_prime=G_prime, eps=eps, delta=delta)
            if l < H_max:
                l = H_max
                f_max = f_

    l = max(0, min(np.log(l), eps))

    return l, f_max
    
