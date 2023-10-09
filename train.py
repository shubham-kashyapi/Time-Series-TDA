import time
import os
import argparse
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view 
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from gtda.homology import VietorisRipsPersistence
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import torch
from torch import nn
from cpd_model import *

def main():
    ###############################################
    # Passing command line arguments
    ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help = 'files containing the datasets', 
                        default = './Datasets/PAMAP2_Dataset/Protocol/subject102.dat', type = str)
    parser.add_argument('--model_save_path', help = 'Path to save the models', 
                        default = './Saved_Models/model_contrastive.pth', type = str)
    parser.add_argument('--plots_save_path', help = 'Path to save the histogram plots during training', 
                        default = './Saved_Models/Plots/', type = str)
    parser.add_argument('--window_len', help = 'Length of window to be used for computing persistence diagrams', 
                        default = 256, type = int)
    parser.add_argument('--hidden_dim', help = 'Output dimension of intermediate MLP layers', 
                        default = 64, type = int)
    parser.add_argument('--num_similar', help = 'Number of similar pairs per class (for contrastive loss)', 
                        default = 50, type = int)
    parser.add_argument('--num_dissimilar', help = '''Number of dissimilar pairs for every pair of
                        adjacent segments (for contrastive loss)''', default = 20, type = int)
    parser.add_argument('--loss_type', help = 'Type of loss function for learning similarity between windows', 
                        default = 'contrastive', choices = ['contrastive', 'triplet'], type = str)
    parser.add_argument('--train_ratio', help = 'Fraction of pairs to use for training',
                        default = 0.6, type = float)
    parser.add_argument('--num_epochs', help = 'Number of epochs to train for',
                        default = 1000, type = int)
    parser.add_argument('--learning_rate', help = 'Learning rate for training',
                        default = 5e-3, type = float)
    parser.add_argument('--use_cuda', help = 'Whether to use GPUs for training', 
                        action = 'store_false') # Use GPUs by default 
   
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    torch_device = torch.device('cuda') if use_cuda else torch.device('cpu')
    ################################################
    # Reading and preprocessing the data
    ################################################
    print('Reading and preprocessing starts')
    arr = np.loadtxt(args.dataset_path)
    col_name_map = {0: 'timestamp (s)', 1: 'activityID', 2: 'heart rate (bpm)', 4: 'hand_acc1', 5: 'hand_acc2', 6: 'hand_acc3',
                    21: 'chest_acc1', 22: 'chest_acc2', 23: 'chest_acc3', 38: 'ankle_acc1', 39: 'ankle_acc2', 40: 'ankle_acc3'}
    df_timeseries = pd.DataFrame.from_dict({col_name_map[col_num]: arr[:, col_num] 
                                            for col_num in col_name_map})
    temp_arr = np.array(df_timeseries.iloc[:, 3:])
    valid_row_ind = ((~np.isnan(temp_arr).any(axis = 1)) & (df_timeseries['activityID'] != 0))
    df_ts_clean = df_timeseries.loc[valid_row_ind, :] 
    df_ts_clean.set_index(np.arange(len(df_ts_clean)), inplace = True)
    df_ts_clean = df_ts_clean.astype({'activityID': int})
                        
    break_pts_orig = [0] + [ind for ind in range(1, len(df_ts_clean)) if (df_ts_clean.iloc[ind, 1] 
                        != df_ts_clean.iloc[ind-1, 1])] + [len(df_ts_clean)]
    intervals_labeled = [([break_pts_orig[ind], break_pts_orig[ind+1]], df_ts_clean.iloc[break_pts_orig[ind], 1]) 
                         for ind in range(len(break_pts_orig)-1)]
    Xdata_cols = np.array(df_ts_clean.iloc[:, 3:])
    ydata_cols = np.array(df_ts_clean['activityID'])
    print('Reading and preprocessing done\n')
    
    ##############################################################
    # Dividing into windows and computing persistence diagrams
    ##############################################################
    print('Persistence diagram computation starts')
    Xdata_arr, ylabels_arr = [], []
    for interval in intervals_labeled:
        window_inds = np.arange(interval[0][0], interval[0][1], args.window_len)
        window_start_end = sliding_window_view(window_inds, 2)
        for window_row in window_start_end:
            Xdata_arr.append(Xdata_cols[window_row[0]: window_row[1], :])
            ylabels_arr.append(interval[1])
    ylabels_arr = np.array(ylabels_arr)
    
    # Sequential computation of persistence diagrams
    homology_dim = [0, 1]
#     Xdata_homology = []
#     for X_arr in tqdm(Xdata_arr):
#         X_arr_homo = VietorisRipsPersistence(homology_dimensions = homology_dim).\
#                          fit_transform(X_arr[None, :, :])[0, :, :]
#         Xdata_homology.append(X_arr_homo)
        
    def get_persistence_diagram(X_arr, homology_dim = homology_dim):
        X_arr_homo = VietorisRipsPersistence(homology_dimensions = homology_dim).\
                         fit_transform(X_arr[None, :, :])[0, :, :]
        return X_arr_homo

    start_time = time.time()    
    Xdata_homology = Parallel(n_jobs = -1)(delayed(get_persistence_diagram)(X_arr) 
                         for X_arr in Xdata_arr)
    end_time = time.time()
    print('Persistence diagram computation done. Time taken = {}\n'.format(end_time-start_time))
    
    #############################################################
    # Preparing the training data (similar and dissimilar pairs)
    #############################################################
    print('Data preparation starts')
    def get_intervals(arr):
        break_pts = [0] + [ind for ind in range(1, len(arr)) if (arr[ind] != arr[ind-1])] + [arr.shape[0]]
        break_start_end = sliding_window_view(break_pts, 2)
        return break_start_end
    seg_intervals = get_intervals(ylabels_arr)
    seg_labels = np.array([ylabels_arr[interval[0]] for interval in seg_intervals])
    complete_dataset = PersisDiagDataset(np.array(Xdata_homology, dtype = object), ylabels_arr)
    # Sampling similar and dissimilar pairs
    similar_pairs = []
    for interval in seg_intervals:
        seg_rand_inds = np.random.randint(interval[0], high = interval[1]-1, size = (args.num_similar, 1))
        seg_similar_adj = np.concatenate((seg_rand_inds, seg_rand_inds+1), axis = 1)
        similar_pairs.append(seg_similar_adj)

    similar_pairs = np.concatenate(similar_pairs, axis = 0)
    similar_pairs = torch.tensor(similar_pairs).to(torch_device)
    
    # Sampling dissimilar pair indices
    dissimilar_pairs = []
    for ind1 in range(len(seg_intervals)):
        for ind2 in range(ind1+1, len(seg_intervals)):
            if seg_labels[ind1] != seg_labels[ind2]:
                pairs_col0 = np.random.randint(seg_intervals[ind1, 0], high = seg_intervals[ind1, 1], 
                                               size = (args.num_dissimilar, 1))
                pairs_col1 = np.random.randint(seg_intervals[ind2, 0], high = seg_intervals[ind2, 1], 
                                               size = (args.num_dissimilar, 1))
                dissimilar_pairs.append(np.concatenate([pairs_col0, pairs_col1], axis = 1))

    dissimilar_pairs = np.concatenate(dissimilar_pairs, axis = 0)
    dissimilar_pairs = torch.tensor(dissimilar_pairs).to(torch_device)
    
    print('Similar pairs = {}, dissimilar pairs = {}'.format(similar_pairs.shape, dissimilar_pairs.shape))
    
    window_pairs = torch.cat([similar_pairs, dissimilar_pairs], dim = 0)
    dissim_labels = torch.cat([torch.zeros(similar_pairs.shape[0]),
                               torch.ones(dissimilar_pairs.shape[0])], dim = 0).to(torch.long).to(torch_device)
    
    # Divide pairs into train and test sets
    tot_windows = window_pairs.shape[0]
    permuted = torch.randperm(tot_windows)
    window_pairs_train = window_pairs[permuted[:int(args.train_ratio*tot_windows)], :]
    dissim_labels_train = dissim_labels[permuted[:int(args.train_ratio*tot_windows)]]
    #print(dissim_labels_train)
    window_pairs_test = window_pairs[permuted[int(args.train_ratio*tot_windows):], :]
    dissim_labels_test = dissim_labels[permuted[int(args.train_ratio*tot_windows):]]
    #print(dissim_labels_test)
    print('Data preparation done\n')
    
    ########################################################
    # Initializing and training the model
    ########################################################
    disc_model = Disc_Model(128, 128, 128, 128, 128) 
    if use_cuda:
        disc_model.cuda()
    optimizer = torch.optim.Adam(list(disc_model.parameters()), lr = args.learning_rate)
    for epoch in range(args.num_epochs):
        print('__________________________________________')
        print('Epoch = {}'.format(epoch))
        ##################
        # Training
        ##################
        dists_train = train_one_epoch_contr(disc_model, complete_dataset, window_pairs_train,
                        dissim_labels_train, optimizer)
        dists_train = dists_train.cpu().detach().numpy()
        labels_train = dissim_labels_train.cpu().detach().numpy()
        if epoch % 50 == 0:
            # Generating plot
            fig = plt.figure()
            #plt.xlim([0.0, 10.0])
            plt.hist(dists_train[labels_train == 0], label = 'similar', bins = 100, density = True)
            plt.hist(dists_train[labels_train == 1], label = 'dissimilar', bins = 100, alpha = 0.5, density = True)
            plt.legend()
            plt.title('Train')
            plt.savefig(os.path.join(args.plots_save_path, 'train_epoch_{}.png'.format(epoch)))
            plt.close(fig)
        ##################
        # Evaluation 
        ##################
        dists_test = eval_one_epoch_contr(disc_model, complete_dataset, window_pairs_test,
                       dissim_labels_test)
        dists_test = dists_test.cpu().detach().numpy()
        labels_test = dissim_labels_test.cpu().numpy()
        if epoch % 50 == 0:
            # Generating plot
            fig = plt.figure()
            #plt.xlim([0.0, 10.0])
            plt.hist(dists_test[labels_test == 0], label = 'similar', bins = 100, density = True)
            plt.hist(dists_test[labels_test == 1], label = 'dissimilar', bins = 100, alpha = 0.5, density = True)
            plt.legend()
            plt.title('Test')
            plt.savefig(os.path.join(args.plots_save_path, 'test_epoch_{}.png'.format(epoch)))
            plt.close(fig)
        print('__________________________________________')
        
    torch.save({    'epoch': args.num_epochs,
                    'model_state_dict': disc_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
               },   args.model_save_path)
    
if __name__ == '__main__':
    main()
