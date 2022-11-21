import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gtda.graphs import TransitionGraph
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape, PersistenceEntropy
from scipy.signal import find_peaks
from scipy.sparse.csgraph import shortest_path
import ot

class Segmentation_Persistent_Homology:
    
    def __init__(self, homology_type, distance_metric):
        '''
        Parameters:
        homology_type: 'ordinal_partition' or 'takens_embedding' or 'multivariate'
        point_summary_type: 'max_persistence', 'periodicity_score', 
                            'homology_classes_by_graph_order', 'norm_persistent_entropy'
                            
        distance_metric (str): Specify the distance metric for quantifying the dissimilarity between
        the persistence diagrams of adjacemt segments. Can take the following values:
        'kernel_dist', 'sliced_wasserstein', 'persistence_landscape'
        
        Returns:
        None
        '''
        valid_homology_types = {'ordinal_partition', 'takens_embedding', 'multivariate'}
        valid_distance_metrics = {'kernel_dist', 'sliced_wasserstein', 'persistence_landscape'}
        if homology_type not in valid_homology_types:
            raise ValueError("Homology type '{}' is invalid".format(homology_type))
        self.homology_type = homology_type
        if distance_metric not in valid_distance_metrics:
            raise ValueError("Distance metric '{}' is invalid".format(distance_metric))
        self.distance_metric = distance_metric
        self.distance_metric_map = {'kernel_dist': self.kernel_dist, 'sliced_wasserstein': self.sliced_wasserstein_dist,
                                    'persistence_landscape': self.persistence_landscape_dist}        
        return
    
    def kernel_dist(self, persis1, persis2, sig = 1.0):
        kernel_func_12, kernel_func_11, kernel_func_22 = 0.0, 0.0, 0.0
        for pt1 in range(persis1.shape[1]):
            for pt2 in range(persis2.shape[1]):
                dist_1 = (persis1[0, pt1, 0]-persis2[0, pt2, 0])**2 + (persis1[0, pt1, 1]-persis2[0, pt2, 1])**2
                dist_2 = (persis1[0, pt1, 0]-persis2[0, pt2, 1])**2 + (persis1[0, pt1, 1]-persis2[0, pt2, 0])**2
                kernel_func_12 = kernel_func_12 + np.exp(-1.0/(8*sig)*dist_1) + np.exp(-1.0/(8*sig)*dist_2)        
        kernel_func_12 = 1.0/(8.0*np.pi*sig)*kernel_func_12
        
        for pt1 in range(persis1.shape[1]):
            for pt2 in range(persis1.shape[1]):
                dist_1 = (persis1[0, pt1, 0]-persis1[0, pt2, 0])**2 + (persis1[0, pt1, 1]-persis1[0, pt2, 1])**2
                dist_2 = (persis1[0, pt1, 0]-persis1[0, pt2, 1])**2 + (persis1[0, pt1, 1]-persis1[0, pt2, 0])**2
                kernel_func_11 = kernel_func_11 + np.exp(-1.0/(8*sig)*dist_1) + np.exp(-1.0/(8*sig)*dist_2)
        kernel_func_11 = 1.0/(8.0*np.pi*sig)*kernel_func_11
                
        for pt1 in range(persis2.shape[1]):
            for pt2 in range(persis2.shape[1]):
                dist_1 = (persis2[0, pt1, 0]-persis2[0, pt2, 0])**2 + (persis2[0, pt1, 1]-persis2[0, pt2, 1])**2
                dist_2 = (persis2[0, pt1, 0]-persis2[0, pt2, 1])**2 + (persis2[0, pt1, 1]-persis2[0, pt2, 0])**2
                kernel_func_22 = kernel_func_22 + np.exp(-1.0/(8*sig)*dist_1) + np.exp(-1.0/(8*sig)*dist_2)
        kernel_func_22 = 1.0/(8.0*np.pi*sig)*kernel_func_22
        
        dist_ker = (kernel_func_11 + kernel_func_22 -2*kernel_func_12)**0.5
        return dist_ker
    
    def sliced_wasserstein_dist(self, persis1, persis2, homology_dim = 1):
        '''
        Parameters:
        persis1, persis2- 2d numpy arrays. Points in the persistence diagram
        
        Returns:
        sliced_wass_dist- float
        '''
        persis1_filtered = persis1[persis1[:, 2] == homology_dim, 0:2]
        persis2_filtered = persis2[persis2[:, 2] == homology_dim, 0:2]
        sliced_wass_dist = ot.sliced.sliced_wasserstein_distance(persis1_filtered, persis2_filtered)
        return sliced_wass_dist
    
    def persistence_landscape_dist(self, persis1, persis2, homology_dim = 1):
        persisLandObj1 = PersistenceLandscape()
        persisLandArr1 = persisLandObj1.fit_transform_plot(persis1)
        persisLandObj2 = PersistenceLandscape()
        persisLandArr2 = persisLandObj2.fit_transform_plot(persis2)
        l2_dist = (1.0/persisLandArr1.shape[2])*np.sum((persisLandArr1-persisLandArr2)**2) # L2 distance between the two persistence landscapes
        return l2_dist
        
    
    def get_persistence_diagram(self, ts_window):
        '''
        Performs Takens Embedding to get the point cloud for a univariate time series.c
        Then computes the persistence diagram for the same.
        
        Parameters:
        ts_window: 1d numpy array (for ordinal_partition) ; 1d or 2d numpy array (for takens_embedding)
        
        Returns:
        None
        '''
        homology_dimensions = [1]
        
        if self.homology_type == 'ordinal_partition':           
            tau_curr, d_curr = 1, 5 # Hard-coded values (TO FIX later)
            indices = np.array([np.arange(i, i+d_curr*tau_curr, tau_curr) 
                                for i in range(ts_window.shape[0]-(d_curr-1)*tau_curr)])
            X_perm_window = ts_window[indices]
            X_tg = TransitionGraph().fit_transform([X_perm_window])[0]
            dist_matrix = shortest_path(X_tg)
            persis = VietorisRipsPersistence(metric = 'precomputed', homology_dimensions = homology_dimensions, 
                                                n_jobs = -1).fit_transform([dist_matrix])
            persis[np.isinf(persis)] = 1e2
            
        elif self.homology_type == 'takens_embedding':
            max_embedding_dimension, stride = 5, 1 # Hard-coded values (TO FIX later)
            max_time_delay = ts_window.shape[0]//(max_embedding_dimension + 1)

            TS_embedded = SingleTakensEmbedding(parameters_type= 'search', time_delay = max_time_delay, 
                                                dimension = max_embedding_dimension,
                                                stride = stride).fit_transform(ts_window)
            
            crshp_embed = TS_embedded[None, :, :]
            
            persis = VietorisRipsPersistence(homology_dimensions = homology_dimensions,
                                             n_jobs = -1).fit_transform(crshp_embed)
            persis[np.isinf(persis)] = 1e2
            
        elif self.homology_type == 'multivariate':
            persis = VietorisRipsPersistence(homology_dimensions = homology_dimensions,
                                             n_jobs = -1).fit_transform(ts_window.reshape(1, *ts_window.shape))
            
        return persis
            
    
    def get_segmentation(self, ts_data, window_len, threshold = 1.0):
        '''
        Parameters:
        ts_data: 1d numpy array (for ordinal_partition) ; 1d or n-dim numpy array (for takens_embedding)
        window_len: integer
        
        Returns:
        None
        '''
        intervals = [[start_ind, start_ind + window_len] for start_ind in range(0, ts_data.shape[0], window_len)]
        intervals[-1][-1] = ts_data.shape[0]
        print(intervals)
        passes_count = 0
        
        while True:
            print('Pass number = {}'.format(passes_count))
            print(intervals)
            
            persisDiagrams = []
            for interval in intervals:
                persisDiagrams.append(self.get_persistence_diagram(ts_data[interval[0]: interval[1]])[0, :, :])
                
            intervals_merged = []
            #print(persisDiagrams[0].shape)
            
            ind = 0
            while ind < len(intervals)-1:
                # Calculate the discrepancy / distance between topological representations of adjacent segments
                dist_adj = self.distance_metric_map[self.distance_metric](persisDiagrams[ind], persisDiagrams[ind+1])
                print('Indices = {} and {}, distance = {}'.format(ind, ind+1, dist_adj))
                if(dist_adj < threshold):
                    intervals_merged.append([intervals[ind][0], intervals[ind+1][1]])
                    ind += 2
                else:
                    intervals_merged.append([intervals[ind][0], intervals[ind][1]])
                    ind += 1
                    
            if ind == len(intervals)-1:
                intervals_merged.append([intervals[ind][0], intervals[ind][1]])
                
            if len(intervals) == len(intervals_merged):
                break
            
            intervals = intervals_merged
            
            passes_count += 1
        
        if self.homology_type != 'multivariate':
            plt.figure()
            plt.scatter(np.arange(ts_data.shape[0]), ts_data, s = 1)
            plt.axvline(x = 0, c = 'r')
            for interval in intervals:
                plt.axvline(x = interval[1], c = 'r')
            plt.show()
        else:
            for interval_ in intervals:
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111, projection = '3d')
                ax.scatter3D(ts_data[interval_[0]:interval_[1], 0], ts_data[interval_[0]:interval_[1], 1], 
                             ts_data[interval_[0]:interval_[1], 2], s = 1)
                ax.set_xlabel('X1')
                ax.set_ylabel('X2')
                ax.set_zlabel('X3')
                ax.set_title('Window = {} to {}'.format(interval_[0], interval_[1]))
        
        return intervals
    
def get_persistence_diagrams(ts_arr, ax, homology_dimensions = [0, 1]):
    max_embedding_dimension, stride = 5, 1 # Hard-coded values (TO FIX later)
    max_time_delay = ts_arr.shape[0]//(max_embedding_dimension + 1)

    TS_embedded = SingleTakensEmbedding(parameters_type= 'search', time_delay = 1, 
                                        dimension = 3, stride = stride).fit_transform(ts_arr)

    crshp_embed = TS_embedded[None, :, :]

    persis = VietorisRipsPersistence(homology_dimensions = homology_dimensions,
                                        n_jobs = -1).fit_transform(crshp_embed)    
    colors = ['b', 'tab:orange', 'g']
    plt.sca(ax)
    for homology_dim, c in zip(homology_dimensions, colors):
        persis_dim = persis[0, :, :]
        persis_dim = persis_dim[persis_dim[:, 2] == homology_dim, :]        
        plt.scatter(persis_dim[:, 0], persis_dim[:, 1], c = c)
    
    plt.scatter(np.linspace(0, np.max(persis_dim[:, 1]), 1000), np.linspace(0, np.max(persis_dim[:, 1]), 1000), c = 'k', s = 1)
    plt.legend(["H{}".format(homology_dim) for homology_dim in homology_dimensions])
    #print(persis)
    return persis

def get_persistence_diagrams_multivariate(ts_arr, ax = None, homology_dimensions = [0, 1], show_plots = False):
    persis = VietorisRipsPersistence(homology_dimensions = homology_dimensions,
                                        n_jobs = -1).fit_transform(ts_arr.reshape(1, *ts_arr.shape))    
    if show_plots:
        colors = ['b', 'tab:orange', 'g']
        plt.sca(ax)
        for homology_dim, c in zip(homology_dimensions, colors):
            persis_dim = persis[0, :, :]
            persis_dim = persis_dim[persis_dim[:, 2] == homology_dim, :]
            plt.scatter(persis_dim[:, 0], persis_dim[:, 1], c = c)


        plt.scatter(np.linspace(0, np.max(persis_dim[:, 1]), 1000), np.linspace(0, np.max(persis_dim[:, 1]), 1000), c = 'k', s = 1)
        plt.legend(["H{}".format(homology_dim) for homology_dim in homology_dimensions])
    #print(persis)
    return persis[0, :, :]

def sliced_wasserstein_dist(persis1, persis2, homology_dim = 1):
    '''
    Parameters:
    persis1, persis2- 2d numpy arrays. Points in the persistence diagram

    Returns:
    sliced_wass_dist- float
    '''
    persis1_filtered = persis1[persis1[:, 2] == homology_dim, 0:2]
    persis2_filtered = persis2[persis2[:, 2] == homology_dim, 0:2]
    sliced_wass_dist = ot.sliced.sliced_wasserstein_distance(persis1_filtered, persis2_filtered)
    return sliced_wass_dist    