'''
    Statistical privacy leakage calculation.

    This class supports multi-thread computation to speed up calculations.
'''

import numpy as np
import pandas as pd
import math
import time 
from utils.util_functions import list_to_string
from concurrent.futures import ProcessPoolExecutor

def get_final_alphabet_list(alphabet_list, base_string = "", output_list = []):
    if len(alphabet_list) == 1:
        for i in alphabet_list[0]:
            if base_string == "":
                output_list.append(str(i))
            else:
                output_list.append(base_string + " " + str(i))
        return output_list
    new_output_list = output_list
    new_base_string = base_string
    for i in alphabet_list[0]:
        if base_string == "":
            new_base_string = str(i)
        else:
            new_base_string = base_string + " " + str(i)
        new_output_list = get_final_alphabet_list(alphabet_list[1:], base_string = new_base_string, output_list = new_output_list)
    return new_output_list

def ignore_nan(arr, eps):
    try:
        return min(max(filter(lambda x: not (math.isinf(x) or math.isnan(x)), arr)), eps)
    except:
        return 0

def cal_empirical_leakage(filtered_original_data, filtered_perturb_data, Z_alphabet, X_k_alphabet, samples, eps):
    probability_matrix = np.zeros((len(X_k_alphabet), len(Z_alphabet)))

    index_x_k = 0
    index_z = 0
    len_of_original_dataset = np.shape(filtered_original_data)[0]
    
    for index_i, i in enumerate(filtered_perturb_data[:samples,:]):
        index_z = Z_alphabet.index(list_to_string(i)[:-1])
        index_x_k = X_k_alphabet.index(list_to_string(filtered_original_data[index_i%len_of_original_dataset])[:-1])
        probability_matrix[index_x_k][index_z] += 1
        
    probability_matrix /= np.nansum(probability_matrix)
    
    conditional_matrix = np.transpose(np.transpose(probability_matrix)/np.nansum(probability_matrix, axis=1))
    max_array = np.nanmax(conditional_matrix, axis=0)
    masked = np.where(conditional_matrix > 0, conditional_matrix, np.inf)
    min_array = np.nanmin(masked, axis=0)
    
    leakage_list = np.array(np.log(max_array/min_array))
    leakage = ignore_nan(leakage_list, eps)
    return leakage

def one_surrogate(parameters):
    filtered_perturb_data, filtered_original_data, num_column, Z_alphabet, X_k_alphabet, sample, eps = parameters

    surrogate_dist = np.zeros(np.shape(filtered_perturb_data))
    for col in range(num_column):
        np.random.seed(int((time.time()*1000000)%1000000))
        surrogate_dist[:,col] = np.random.permutation(filtered_perturb_data[:,col])
    surrogate_dist = np.array(surrogate_dist, dtype=int)
    S_sur = cal_empirical_leakage(filtered_original_data=filtered_original_data, filtered_perturb_data=surrogate_dist, Z_alphabet=Z_alphabet, X_k_alphabet=X_k_alphabet, samples=sample, eps=eps)
    return S_sur
    
def get_p_val(filtered_original_data, filtered_perturb_data, Z_alphabet, X_k_alphabet, actual_leakage, num_surrogates, eps, num_processes = 10, samples = 200000):
    surrogate_stats = []

    num_column = np.shape(filtered_perturb_data)[1]

    for i in range(num_surrogates//num_processes):
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(one_surrogate, (filtered_perturb_data, filtered_original_data, num_column, Z_alphabet, X_k_alphabet, samples, eps)) for j in range(num_processes)]
            results_ = [future.result() for future in futures]
        for j in results_:
            surrogate_stats.append(j)

    surrogate_stats = np.array(surrogate_stats)
    p_value = np.mean(np.abs(surrogate_stats) >= np.abs(actual_leakage))

    return p_value


class Empirical_Privacy():
    
    def __init__(self, original_data, perturb_data):
        self.original_data = original_data
        self.perturb_data = perturb_data

    def total_privacy_leakage(self, Z, X_k, eps, p_value_cal = False, num_surrogates = 100, samples = 200000):
        Z_copy = list(Z)
        Z_copy.append(X_k[0])
        return self.additional_privacy_leakage(Z_copy, X_k, eps, p_value_cal, num_surrogates, samples = samples)
       
    def cal_cpl(self, Z, X_k, eps, p_value_cal = False, num_surrogates = 100, samples = 200000):
        self.X_k_alphabet = self.get_ordered_alphabet_(X_k)
        self.Z_alphabet = self.get_ordered_alphabet_(Z)

        filtered_original_data = self.original_data[X_k].values
        filtered_perturb_data = self.perturb_data[Z].values
        samples = len(filtered_perturb_data)
        actual_leakage = cal_empirical_leakage(filtered_original_data= filtered_original_data, filtered_perturb_data=filtered_perturb_data, Z_alphabet= self.Z_alphabet, X_k_alphabet=self.X_k_alphabet, samples=samples, eps=eps)

        p_val = None
        if p_value_cal:
            p_val = get_p_val(filtered_original_data= filtered_original_data, filtered_perturb_data=filtered_perturb_data, Z_alphabet= self.Z_alphabet, X_k_alphabet=self.X_k_alphabet, actual_leakage=actual_leakage, num_surrogates = num_surrogates, eps = eps, samples=samples)

        return actual_leakage, p_val

    def get_ordered_alphabet_(self, attributes):
        alphabet_list = []

        if not(isinstance(attributes, list)):
            attributes = [attributes]

        for i in attributes:
            alphabet_list.append(self.get_ordered_alphabet_individual(i))

        return (get_final_alphabet_list(alphabet_list=alphabet_list, base_string = "", output_list = []))
        
    def get_ordered_alphabet_individual(self, attribute):
        return (np.unique(self.original_data[attribute].values))
    
