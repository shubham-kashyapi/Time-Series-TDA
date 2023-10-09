import time
import os
import argparse
from joblib import Parallel, delayed
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view 
from scipy.signal import find_peaks
import pandas as pd
from tqdm import tqdm
from gtda.homology import VietorisRipsPersistence
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from torch import nn
from cpd_model import *

def preprocess_timeseries(file_path):
    ###############################
    # Loading the data
    ###############################
    arr = np.loadtxt(file_path)
    col_name_map = {0: 'timestamp (s)', 1: 'activityID', 2: 'heart rate (bpm)', 4: 'hand_acc1', 
                    5: 'hand_acc2', 6: 'hand_acc3', 21: 'chest_acc1', 22: 'chest_acc2', 
                    23: 'chest_acc3', 38: 'ankle_acc1', 39: 'ankle_acc2', 40: 'ankle_acc3'}
    df_timeseries = pd.DataFrame.from_dict({col_name_map[col_num]: arr[:, col_num] 
                                            for col_num in col_name_map})
    
    ################################
    # Removing missing data 
    ################################
    temp_arr = np.array(df_timeseries.iloc[:, 3:])
    valid_row_ind = ((~np.isnan(temp_arr).any(axis = 1)) & (df_timeseries['activityID'] != 0))
    df_ts_clean = df_timeseries.loc[valid_row_ind, :] 
    df_ts_clean.set_index(np.arange(len(df_ts_clean)), inplace = True)
    df_ts_clean = df_ts_clean.astype({'activityID': int})
    
    ################################
    # Retrieving the change points
    ################################
    change_pts = [0] + [ind for ind in range(1, len(df_ts_clean)) 
                        if (df_ts_clean.iloc[ind, 1] != df_ts_clean.iloc[ind-1, 1])] + \
                        [len(df_ts_clean)]
    Xdata_cols = np.array(df_ts_clean.iloc[:, 3:])
    
    ################################
    # Retrieving the class labels
    ################################
    cls_segments = [] # Classwise segments
    for time_ind in range(len(change_pts)-1):
        cls_segments.append(((change_pts[time_ind], change_pts[time_ind+1]), 
                              df_ts_clean.loc[change_pts[time_ind], 'activityID']))
    
    
    ################################
    print(change_pts, '\n', cls_segments)
    return Xdata_cols, change_pts, cls_segments


def pipeline(filepath, dataset_name, window_len = 256, offset = 256,
             torch_device = torch.device('cuda:0')):
    #####################################
    # Read and preprocess data
    #####################################
    print('_____________________________________________')
    print(filepath)
    start_time = time.time()
    Xdata_cols, change_pts, cls_segments = preprocess_timeseries(filepath)
    end_time = time.time()
#     print(dataset_name)
#     print(Xdata_cols.shape)
#     print(change_pts)
    print('Preprocessing = {}'.format(end_time-start_time))
    ########################################################
    # Divide into windows and compute persistence diagrams
    ########################################################
    start_time = time.time() 
    # windows_cpd = sliding_window_view(np.arange(0, Xdata_cols.shape[0], window_len), 2)
    windows_left_ind = np.arange(0, Xdata_cols.shape[0], offset).reshape(-1, 1)
    windows_right_ind = windows_left_ind + window_len
    windows_cpd = np.concatenate((windows_left_ind, windows_right_ind), axis = 1)
    Xwindow_arr = []
    for interval in windows_cpd:
        Xwindow_arr.append(Xdata_cols[interval[0]: interval[1], :]) 
        
    def get_persistence_diagram(X_arr, homology_dim = [0, 1]):
        X_arr_homo = VietorisRipsPersistence(homology_dimensions = homology_dim).\
                         fit_transform(X_arr[None, :, :])[0, :, :]
        return X_arr_homo
    
    Xcpd_homology = Parallel(n_jobs = -1)(delayed(get_persistence_diagram)(X_arr) 
                         for X_arr in Xwindow_arr)
    cpd_dataset = PersisDiagDataset(np.array(Xcpd_homology, dtype = object), None)
    end_time = time.time()
    print('Windows and persistence diagrams = {}'.format(end_time-start_time))
    
    #######################################################
    # Computing the change points
    #######################################################
    start_time = time.time() 
    
    disc_model_saved = Disc_Model(128, 128, 128, 128, 128)
    disc_model_saved.load_state_dict(torch.load('./Saved_Models/model_contrastive.pth')['model_state_dict'])
    disc_model_saved.to(torch_device)
    #print(disc_model_saved.device)
    
    inds_pair0 = torch.arange(0, len(cpd_dataset)-1).to(torch_device)
    inds_pair1 = torch.arange(1, len(cpd_dataset)).to(torch_device)
    
    diagrams1, diagrams2 = cpd_dataset[inds_pair0], cpd_dataset[inds_pair1]
    embeds1, embeds2 = disc_model_saved(diagrams1), disc_model_saved(diagrams2)
    dists_adj = torch.norm(embeds1-embeds2, dim = 1)
#     print(dists_adj.device)
    
    def moving_avg(arr, w):
        mvg_avg = np.zeros((arr.shape[0],))
        for ind in range(arr.shape[0]):
            mvg_avg[ind] = np.mean(arr[ind: ind+w])
        return mvg_avg
    
    for w in [5, 10, 20]: # Moving average window length
        dists_moving_avg = moving_avg(dists_adj.detach().cpu().numpy(), w)
        # Plotting true change points with moving avg of distances
        fig = plt.figure()
        plt.plot(windows_cpd[:-1, 1], dists_moving_avg)
        for pt in change_pts:
            plt.axvline(pt, c = 'r', lw = 1.0)
        plt.grid()
        #plt.ylim([0, 200])
        plt.title('Moving avg window = {w}'.format(w = w))
        plt.savefig('./Model_Evaluation/{}_movavg{}.png'.format(dataset_name, w))
        plt.close(fig)
    end_time = time.time()
    print('CPD computation = {}'.format(end_time-start_time))
    
    #######################################################
    # TSNE plots for all windows (based on pairwise
    # similarity computed by the discriminator model)
    #######################################################
    # Gettting labelled windows
    windows_classwise, labels_classwise = [], []
    for segment in cls_segments:
        start_ind, end_ind = segment[0][0], segment[0][1]
        windows_start = np.arange(start_ind, end_ind-window_len, offset)
        windows_cls = np.stack((windows_start, windows_start+window_len), axis = 1)
        windows_classwise.append(windows_cls)
        labels_cls = np.array([segment[1]]*windows_cls.shape[0])
        labels_classwise.append(labels_cls)
    windows_classwise = np.concatenate(windows_classwise, axis = 0)
    labels_classwise = np.concatenate(labels_classwise, axis = 0)
#     print(windows_classwise.shape, labels_classwise.shape)
#     print(windows_classwise[:10, :], '\n', labels_classwise[:10])
#     print(windows_classwise[-10:, :], '\n', labels_classwise[-10:])
    
    # Computing persistence diagrams of the windows
    Xwindow_cls_arr = []
    for interval_cls in windows_classwise:
        Xwindow_cls_arr.append(Xdata_cols[interval_cls[0]: interval_cls[1], :]) 
    
    Xcls_persis = Parallel(n_jobs = -1)(delayed(get_persistence_diagram)(X_arr) 
                         for X_arr in Xwindow_cls_arr)
#     print('______________________________________________')
#     print(len(Xcls_persis), Xcls_persis[0].shape, '\n', Xcls_persis[0][:10, :])
#     print('______________________________________________')
    cls_dataset = PersisDiagDataset(np.array(Xcls_persis, dtype = object), None) 
    diags_cls_all = cls_dataset[:]
#     print(len(cls_dataset))
    
    # Computing the distance metric for each pair of diagrams 
    diags_embeds = disc_model_saved(diags_cls_all)
    inds_pairs_all = torch.tensor([[ind0, ind1] for ind0 in range(len(cls_dataset))
                                   for ind1 in range(len(cls_dataset))]).to(torch.long).to(torch_device)
#     print(inds_pairs_all.shape)
#     print(inds_pairs_all[:10, :], '\n', inds_pairs_all[-10:, :])
    embeds1, embeds2 = diags_embeds[inds_pairs_all[:, 0]], diags_embeds[inds_pairs_all[:, 1]]
    dists_pairwise = torch.norm(embeds1-embeds2, dim = 1)
    dists_pairwise = torch.reshape(dists_pairwise, (len(cls_dataset), len(cls_dataset))).detach().cpu().numpy()
    print(dists_pairwise.shape)
    
    # Obtaining TSNE embeddings
    tsne_obj = TSNE(n_components = 2, metric = 'precomputed', init = 'random', random_state = 100)
    tsne_embed = tsne_obj.fit_transform(dists_pairwise)
    
    # Generating plot with activity-wise colouring
    activity_unique_labels = [4, 5, 6, 7, 12, 13]
    activity_label_ids = list(np.arange(len(activity_unique_labels)))
    activity_label_color_ids = (1.0/(np.max(activity_label_ids)-
                                np.min(activity_label_ids)))*(activity_label_ids-np.min(activity_label_ids))
    cm = plt.get_cmap('rainbow')
    fig = plt.figure(figsize = (20, 12))
    for activity_id_, activity_color_id_ in zip(activity_unique_labels, activity_label_color_ids):
        activity_indices = (np.array(labels_classwise) == activity_id_)
        plt.scatter(tsne_embed[activity_indices, 0], tsne_embed[activity_indices, 1], color = cm(activity_color_id_),
                    label = 'Activity {}'.format(activity_id_))
    plt.legend()
    plt.grid()
    plt.title(dataset_name)
    plt.savefig('./Model_Evaluation/TSNE_Modified1_{}.png'.format(dataset_name))
    plt.close(fig)
    
    
    print('_____________________________________________')

def main():
    dir_path = './Datasets/PAMAP2_Dataset/Protocol/'
    files_use = ['subject101.dat',
                 'subject102.dat',
                 'subject103.dat',
                 'subject104.dat',
                 'subject105.dat',
                 'subject106.dat',
                 'subject107.dat',
                 'subject108.dat']

    for filename in files_use:
        pipeline(os.path.join(dir_path, filename), os.path.basename(filename))

if __name__ == '__main__':
    main()
    