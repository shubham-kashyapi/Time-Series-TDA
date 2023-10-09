import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from topological_layers import Topo_Signature_Layer
from sklearn.metrics import f1_score

class PersisDiagDataset(Dataset):
    '''Dataset consisting of multiple persistence diagrams
       along with their labels
    '''
    def __init__(self, Xhomol_diags, ylabels, homol_dims = [0, 1], 
                 device = torch.device('cuda:0'), transform = None):
        '''
        Xhomol_diags: np.array of objects. Each object is a persistence diagram (np.array) of size 
                      (num_pts, 3) as returned by giotto-tda. Note that num_pts is different for each 
                      persistence diagram
        ylabels: 1D np.array(dtype = int) or None 
        homol_dims: List or Tuple of int. Homology dimension to use
        '''
        super().__init__()
        self.homol_dims = homol_dims
        self.device = device
        if ylabels is not None:
            self.ylabels = torch.tensor(ylabels).to(device)
        else:
            self.ylabels = None
        self.length = Xhomol_diags.shape[0]
        self.persis = dict()
        self.diagram_indices = dict()
        for dim in self.homol_dims:
            dim_diagrams = [] # All persistence diagrams for dimension = dim
            dim_indices = [] # Start and end indices of persistence diagrams
            start_ind = 0
            for persis_alldim in Xhomol_diags:
                diag_currdim = persis_alldim[persis_alldim[:, 2] == dim, :][:, :2]
                dim_diagrams.append(diag_currdim)
                end_ind = start_ind + diag_currdim.shape[0]
                dim_indices.append([start_ind, end_ind])
                start_ind = end_ind
                
            self.persis[dim] = torch.tensor(np.concatenate(dim_diagrams, axis = 0)).to(device).to(torch.float)
            self.diagram_indices[dim] = torch.tensor(dim_indices).to(device)
    
    def __len__(self):
        '''
        Returns the number of persistence diagrams in a dataset
        '''
        return self.length
        
    def __getitem__(self, idx):
        '''
        Given an index, returns persistence diagrams corresponding to
        each homology dimension
        '''
        #print(list(idx))
        Xpersis_idx = dict()
        diags_idx = dict()
        for dim in self.homol_dims:
            slices = self.diagram_indices[dim][idx]
            sliced_persis = [self.persis[dim][start:end] for start, end in slices]
            Xpersis_idx[dim] = torch.cat(sliced_persis, dim = 0).to(self.device)
            slice_lengths = slices[:, 1]-slices[:, 0]
            slice_cumsum = torch.tensor([0]).to(self.device)
            slice_cumsum = torch.cat([slice_cumsum, torch.cumsum(slice_lengths, 0)])
            diags_idx[dim] = torch.stack([slice_cumsum[:-1], 
                                          slice_cumsum[1:]]).to(self.device).t()
            
        ylabels_idx = self.ylabels[idx] if (self.ylabels is not None) else None
        return {'persis': Xpersis_idx, 'slices': diags_idx, 'labels': ylabels_idx}

class Disc_Model(nn.Module):
    
    def __init__(self, homo_zero_embed, homo_one_embed, embed_dim1, embed_dim2, embed_dim3):
        '''
        homo_zero_embed (int): Embedding dimension for H0 persistence diagrams
        homo_one_embed (int): Embedding dimension for H1 persistence diagrams
        embed_dim1, embed_dim2, embed_dim3 (int): Embedding dimension of intermediate representations 
        of peristence diagram (after combining H0 and H1)
        '''
        super().__init__()
        self.model_homo_zero = Topo_Signature_Layer(homo_zero_embed)
        self.model_homo_one = Topo_Signature_Layer(homo_one_embed)
#         concat_homo_embed = homo_zero_embed + homo_one_embed
        concat_homo_embed = homo_one_embed
        self.model_embed = nn.Sequential(nn.Linear(concat_homo_embed, embed_dim1), nn.ReLU(),
                                         nn.Linear(embed_dim1, embed_dim2), nn.ReLU(),
                                         nn.Linear(embed_dim2, embed_dim3), nn.ReLU())
    
    def forward(self, diags):
        '''
        diags: PersisDiagDataset consisting of multiple persistence diagrams
        
        Returns: diags_embed- (len(diags), ) torch.float tensor
        '''
        # Getting persistence diagrams corresponding to H0 and H1
        diags_homo_0, diags_homo_1 = diags['persis'][0], diags['persis'][1]
        diags_slices_0, diags_slices_1 = diags['slices'][0], diags['slices'][1]
        # Passing data to the topological layer
#         diags1_homol = torch.cat([self.model_homo_zero(diags1_homo_0, diags1_slices_0), \
#                                   self.model_homo_one(diags1_homo_1, diags1_slices_1)], dim = 1)
#         diags2_homol = torch.cat([self.model_homo_zero(diags2_homo_0, diags2_slices_0), \
#                                   self.model_homo_one(diags2_homo_1, diags2_slices_1)], dim = 1)
        diags_homol = self.model_homo_one(diags_homo_1, diags_slices_1)
        # After topological layer, pass to a fully connected layer
        diags_embed = self.model_embed(diags_homol)
        return diags_embed    
    
def contrastive_loss(Xdists, Ylabels, mindist = 1000.0, maxdist = 5.0):
    '''    
    Xdists: (num_pairs, ) float torch.tensor
    Ylabels: (num_pairs, ) int torch.tensor (0/1 only). 0 for similar pairs, 1 for dissimilar pairs.
    mindist: (float). Desired minimum distance for dissimilar pairs
    maxdist: (float). Desired maximum distance for similar pairs
    
    Returns:
    contr_loss (torch.float)- Contrastive loss defined (for a single pair) as
    loss = (1-y)*dist^2 + y*max(0, mindist-dist)^2
    '''
    losses = (1-Ylabels)*Xdists + Ylabels*nn.ReLU()(mindist-Xdists)
    return torch.mean(losses)

def forward_pass_contr(discrim_model, dataset_all_windows, ind_pair0, ind_pair1):
    '''
    dists: (ind_pair0.shape[0], ) Distance for each pair
    '''
    diagrams1, diagrams2 = dataset_all_windows[ind_pair0], dataset_all_windows[ind_pair1]
    embeds1, embeds2 = discrim_model(diagrams1), discrim_model(diagrams2)
    dists = torch.square(torch.norm(embeds1-embeds2, dim = 1))
    return dists

def train_one_epoch_contr(discrim_model, dataset_all_windows, ind_pairs, y_labels, 
                          optimizer, mindist = 1000.0, maxdist = 5.0, threshold = 100.0):
    discrim_model.train()
    optimizer.zero_grad()
    # Forward pass and computining train loss
    X_dists = forward_pass_contr(discrim_model, dataset_all_windows, ind_pairs[:, 0], ind_pairs[:, 1])
    print(X_dists.shape)
    train_loss = contrastive_loss(X_dists, y_labels)
    # Gradient descent
    train_loss.backward()
    optimizer.step()
    # Computing train accuracy
    pred_labels = (X_dists > threshold)
    print(pred_labels)
    train_acc = torch.mean((pred_labels == y_labels).float())
    train_f1 = f1_score(y_labels.cpu(), pred_labels.cpu())
    print('Train Loss = {}, Train Accuracy = {}, Train F1 = {}'.format(train_loss, train_acc, train_f1))
    return X_dists
    

def eval_one_epoch_contr(discrim_model, dataset_all_windows, ind_pairs, y_labels,
                   mindist = 1000.0, maxdist = 5.0, threshold = 20.0):
    discrim_model.eval()
    with torch.no_grad():
        # Forward pass and computining test loss
        X_dists = forward_pass_contr(discrim_model, dataset_all_windows, ind_pairs[:, 0], ind_pairs[:, 1])
        test_loss = contrastive_loss(X_dists, y_labels)
        # Computing test accuracy
        pred_labels = (X_dists > threshold)
        test_acc = torch.mean((pred_labels == y_labels).float())
        test_f1 = f1_score(y_labels.cpu(), pred_labels.cpu())
        
    print('Test loss = {}, Test accuracy = {}, Test f1 = {}'.format(test_loss, test_acc, test_f1))
    return X_dists