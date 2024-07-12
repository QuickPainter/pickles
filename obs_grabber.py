
import subprocess
import sys
import os
# from boundary_checker import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import statistics as stats
import pandas as pd
from scipy import signal
import sys
import datetime
import h5py
from scipy.stats import pearsonr 
import scipy.stats  
from tqdm import tqdm
import traceback
import hdf5plugin
import argparse
import pickle
import math
import gc
from scipy.signal import find_peaks
import itertools
import csv
import threading


def super_wrapper(file,on_threshold,off_threshold):
    k_score_table = pd.read_csv(file,dtype={"drift2": 'boolean',"blip or broadband": 'boolean'})
    
    if 'median_k_score' in k_score_table.columns:
        k_score_table.rename(columns={'median_k_score': 'med_k', 'min_k_score': 'min_k'}, inplace=True)
    else:
        k_score_table['med_k'] = k_score_table[['k1', 'k2', 'k3']].apply(np.median, axis=1)
        k_score_table['min_k'] = k_score_table[['k1', 'k2', 'k3']].apply(np.min, axis=1)
        k_score_table['max_k'] = k_score_table[['k1', 'k2', 'k3']].apply(np.max, axis=1)
        # k_score_table['Batch Info'] = ['(-1,-1)']*len(k_score_table)
        k_score_table['drift2'] = [1]*len(k_score_table)
        k_score_table['drift1'] = [1]*len(k_score_table)
        k_score_table['Block Size'] = [500]*len(k_score_table)

        print(np.array(k_score_table["Batch Info"])[-1])

    batch_info = np.array(k_score_table["Batch Info"])
    num_obs_run_out = eval(batch_info[-1])

        
    all_k_scores = np.array(k_score_table["k_score"])

    k1_values = np.array(k_score_table["k1"])
    k2_values = np.array(k_score_table["k2"])
    k3_values = np.array(k_score_table["k3"])
    k4_values = np.array(k_score_table["k4"])
    k5_values = np.array(k_score_table["k5"])
    k6_values = np.array(k_score_table["k6"])

    med_ks = np.array(k_score_table["med_k"])
    min_ks = np.array(k_score_table["min_k"])
    max_ks = np.array(k_score_table["max_k"])


    k_score_table["new_k"] = med_ks
    k_score_table["off_k_sum"] = k2_values+k4_values+k6_values
    
    high_k_outliers= k_score_table[(k_score_table["new_k"]>on_threshold) & (k_score_table["k2"]<off_threshold) & (k_score_table["k4"]<off_threshold) & (k_score_table["k6"]<off_threshold) & (k_score_table["drift2"] == 1.0) & (k_score_table["drift1"] == 1.0)]
    
    index_diff = high_k_outliers['Index'].diff()
    
    # Filter rows, keeping those where the difference is not 0.5
    high_k_outliers = high_k_outliers[index_diff != 0.5]

    high_k_outliers.head()
    
    return high_k_outliers, (num_obs_run_out[0],num_obs_run_out[1]), len(k_score_table), batch_info,k_score_table


# implement max_range column
def get_bbs(high_k_outliers):
    max_ranges = []
    bbs = []
    for i in tqdm(range(0,len(high_k_outliers))):
        obs1_maxes = eval(np.array(high_k_outliers["obs1 maxes"])[i])
        obs3_maxes = eval(np.array(high_k_outliers["obs3 maxes"])[i])
        obs5_maxes = eval(np.array(high_k_outliers["obs5 maxes"])[i])
        
        snr_1,threshold_1 = get_snr(obs1_maxes,10)
        snr_3,threshold_3 = get_snr(obs3_maxes,10)
        snr_5,threshold_5 = get_snr(obs5_maxes,10)
    
        blip_or_broadband = False
        if sum([snr_1,snr_3,snr_5]) >= 1:
            blip_or_broadband = True
    
        ranges = [np.median(obs1_maxes)-np.max(obs1_maxes),np.median(obs3_maxes)-np.max(obs3_maxes),np.median(obs5_maxes)-np.max(obs5_maxes)]
        max_ranges.append(np.min(ranges))
        bbs.append(blip_or_broadband)
    return bbs
    # except:
    #     bbs = [False]*len(high_k_outliers)
    #     return bbs

    
def find_missing_obs(obs_run):
    all_numbers = set(range(1, 1000))
    contained_numbers = set(obs_run)
    missing_numbers = all_numbers - set(obs_run)
    return sorted(missing_numbers), sorted(contained_numbers)

    
def main():

    contained_all = []
    missing_all = []
    
    batch_range = range(4,101)
    all_outliers = []
    all_batch_infos = []
    k_score_table = 0
    for num,batch in tqdm(enumerate(batch_range)): 
        try:
            print(batch)
            if (batch > 10) and (batch<21):
                outliers, e1,e2, batch_info,k_score_table = super_wrapper(f"/datax/scratch/calebp/k_scores/updated_all_cadences_mason_jar_batch_{batch}_block_size_1024_snr_10_section_False.csv",3,.1)
                all_outliers.extend(outliers.to_dict('records'))
                all_batch_infos.append(batch_info)
    
                unique_batch_numbers = set(list(k_score_table["Batch Info"]))
                obs_run = np.sort([eval(x)[1] for x in unique_batch_numbers])
                missing_obs, contained_numbers = find_missing_obs(obs_run)
                
                contained_all.append(contained_numbers)
                missing_all.append(missing_obs)

                print(f"Batch {batch}, Missing: {len(missing_obs)}, Contained: {len(contained_numbers)}")
                
            else:
                outliers, e1,e2,batch_info,k_score_table = super_wrapper(f"/datax/scratch/calebp/k_scores/updated_all_cadences_mason_jar_batch_{batch}_block_size_1024_snr_10_section_True.csv",3,.1)
                all_outliers.extend(outliers.to_dict('records'))
                all_batch_infos.append(batch_info)
    
                unique_batch_numbers = set(list(k_score_table["Batch Info"]))
                obs_run = np.sort([eval(x)[1] for x in unique_batch_numbers])
                missing_obs, contained_numbers = find_missing_obs(obs_run)
                contained_all.append(contained_numbers)
                missing_all.append(missing_obs)

                
                print(f"Batch {batch}, Missing: {len(missing_obs)}, Contained: {len(contained_numbers)}")


        except Exception:
            contained_all.append(-1)
            missing_all.append(-1)

            print(traceback.print_exc())
            print(f"Table {batch} not Found")

    with open("missing", "wb") as fp:   #Pickling
        pickle.dump(missing_all, fp)
    
    with open("contained", "wb") as fp:   #Pickling
        pickle.dump(contained_all, fp)



if __name__ == '__main__':
    main()