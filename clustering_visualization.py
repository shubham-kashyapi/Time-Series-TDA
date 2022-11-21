import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TDA_Segmentation import *
from joblib import Parallel, delayed

def separate_into_known_segments(df_data, label_column):
    N_df = len(df_data)
    start_ind_, curr_label_ = 0, df_data.loc[0, label_column]
    separated_segs = []
    for ind_ in range(N_df):
        if df_data.loc[ind_, label_column] != curr_label_:
            separated_segs.append((curr_label_, (start_ind_, ind_)))
            curr_label_ = df_data.loc[ind_, label_column]
            start_ind_ = ind_
    
    separated_segs.append((int(curr_label_), (start_ind_, N_df)))
    return separated_segs

def get_samples_from_every_segment(window_len, num_intervals, activity_index_pairs_):
    activity_sampled_intervals = []
    for activity_name_, activity_interval_ in activity_index_pairs_:
        sampled_intervals_start = np.linspace(activity_interval_[0], activity_interval_[1]-window_len, num_intervals, dtype = int)
        sampled_intervals_end = np.linspace(activity_interval_[0]+window_len, activity_interval_[1], num_intervals, dtype = int)
        sampled_intervals = [(int(start_), int(end_)) for start_, end_ in zip(sampled_intervals_start, sampled_intervals_end)]
        activity_sampled_intervals.append((int(activity_name_), sampled_intervals))

    return activity_sampled_intervals

def get_activity_index_pairs(df_data, activity_list, activity_id_col, activity_code_reverse_map):
    separated_seg_data = separate_into_known_segments(df_data, activity_id_col)
    activity_id_list = [int(activity_code_reverse_map[activity_name_]) for activity_name_ in activity_list]
    required_seg_data = [seg_pair for seg_pair in separated_seg_data if (int(seg_pair[0]) in activity_id_list)]
    return required_seg_data

def pairwise_segment_diatance_processing_pipeline(data_file_name):
    ########################################
    # Reading and filtering the dataset
    ########################################
    print("______________________________________________")
    print("Loading and preprocessing starts")
    start_time = time.time()
    arr = np.loadtxt(os.path.join(data_dir_path, data_file_name))
    df_timeseries = pd.DataFrame.from_dict({col_name_map[col_num]: arr[:, int(col_num)] for col_num in col_name_map})
    df_timeseries[activity_id_col] = df_timeseries[activity_id_col].astype(int)
    df_timeseries.dropna(axis = 0, how = 'any', subset = cols_to_use, inplace = True)
    df_timeseries.reset_index(drop = True, inplace = True)

    activity_segs = get_activity_index_pairs(df_timeseries, activities_to_use, activity_id_col, activity_code_reverse_map)
    sampled_intervals = get_samples_from_every_segment(window_len, windows_per_segment, activity_segs)
    sampled_intervals_flattened = [interval_val_ for interval_pair_ in sampled_intervals for interval_val_ in interval_pair_[1]]
    end_time = time.time()
    print("Loading and preprocessing complete. Time taken = {}".format(end_time-start_time))
    print("______________________________________________")

    #####################################################################
    # Computing persistence diagrams for all intervals of each activity
    #####################################################################
    print("______________________________________________")
    print("Persistence diagram computation starts")
    start_time = time.time()
    # Parallelized computation of persistence diagrams    
    persisDiagramsFlattened = Parallel(n_jobs=-1)(delayed(get_persistence_diagrams_multivariate)(
                                                      df_timeseries.loc[interval_val_[0]: interval_val_[1], cols_to_use].values)
                                                  for interval_val_ in sampled_intervals_flattened)

    end_time = time.time()
    print("Persistence diagram computation complete. Time taken = {}".format(end_time-start_time))
    print("______________________________________________")

    #################################################
    # Finding the pairwise dissimilarity matrix
    #################################################
    print("______________________________________________")
    print("Pairwise dissimilarity computation starts\n\n")
    start_time = time.time()
    num_segs = len(activity_segs)
    tot_num_samples = num_segs*windows_per_segment
    index_pairs_for_similarity = [(ind_1, ind_2) for ind_1 in range(tot_num_samples) for ind_2 in range(ind_1, tot_num_samples)]

    # Parallelized computation of sliced Wasserstein distance
    dissimilarity_vals = Parallel(n_jobs=-1)(delayed(sliced_wasserstein_dist)(
                                                 persisDiagramsFlattened[ind_pair[0]], persisDiagramsFlattened[ind_pair[1]]) 
                                             for ind_pair in index_pairs_for_similarity)

    dissimilarity_matrix = np.zeros((tot_num_samples, tot_num_samples), dtype = float)

    for ind_pair, dissimilar_val in zip(index_pairs_for_similarity, dissimilarity_vals):
        dissimilarity_matrix[ind_pair[0], ind_pair[1]] = dissimilar_val
        dissimilarity_matrix[ind_pair[1], ind_pair[0]] = dissimilar_val

    end_time = time.time()
    print("Pairwise dissimilarity computation complete. Time taken = {}".format(end_time-start_time))
    print("______________________________________________")

    ############################################################
    # Saving the sampled intervals and dissimilarity matrix
    ############################################################
    print("______________________________________________")
    print("Saving results starts")
    
    file_base_name = data_file_name.rsplit(".", 1)[0]
    np.savetxt(os.path.join(output_path, "{}-segment_similarity_matrix.csv".format(file_base_name)), 
               dissimilarity_matrix, delimiter = ",")

    with open(os.path.join(output_path, "{}-sampled_intervals.json".format(file_base_name)), "w") as f:
        f.write(json.dumps(sampled_intervals))

    print("Saving results complete")
    print("______________________________________________")
    return

if __name__ == '__main__':
    data_dir_path = sys.argv[1]
    output_path = sys.argv[2]
    json_metadata_path = sys.argv[3]
    with open(json_metadata_path, 'r') as f:
        metadata_dict = json.loads(f.read())

    # Loading the metadata of the dataset
    col_name_map, activity_code_map = metadata_dict['col_name_map'], metadata_dict['activity_code_map']
    activities_to_use, cols_to_use = metadata_dict['activities_to_use'], metadata_dict['cols_to_use']
    activity_id_col = metadata_dict['activity_id_col']
    activity_code_reverse_map = {val_: key_ for key_, val_ in activity_code_map.items()}
    
    window_len, windows_per_segment = int(sys.argv[4]), int(sys.argv[5])
    
    for file_name_ in os.listdir(data_dir_path):
        print("========================================================")
        print("Processing file {}".format(file_name_))
        pairwise_segment_diatance_processing_pipeline(file_name_)
        print("Completed processing the file.")
        print("========================================================\n\n")
    
##################################################
# Command for running the script
# python clustering_visualization.py "./PAMAP2_Dataset/Protocol/" "./Persistence_Homology_Output/" "./dataset_info.json" 1200 100
##################################################

