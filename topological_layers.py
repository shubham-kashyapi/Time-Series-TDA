import torch
from torch import nn
import numpy as np

class Topo_Signature_Layer(nn.Module):
    '''
    Neural network layer for learning features from persistence diagrams
    as described in the paper:
    Hofer, Christoph, et al. "Deep learning with topological signatures." NIPS 2017.
    https://arxiv.org/pdf/1707.04041.pdf
    '''
    
    def __init__(self, num_units, threshold = 0.01, device = torch.device('cuda')):
        '''
        num_units (int): Number of hidden units in the persistence layer
        threshold (float): The nu value as defined in the paper
        '''
        super().__init__()
        self.device = device
        self.num_units = num_units
        self.thresh = threshold
        self.mu0 = nn.Parameter(torch.empty(num_units))
        # Since mu1, sigma0 and sigma1 are restricted to be positive,
        # we optimize their logs and get the actual values of the parameters by exponentiation.
        self.log_mu1 = nn.Parameter(torch.empty(num_units))
        self.log_sigma0 = nn.Parameter(torch.empty(num_units))
        self.log_sigma1 = nn.Parameter(torch.empty(num_units))
        # Initialize the parameters
        for param in self.parameters():
            nn.init.normal_(param)
        # Defining the rotation matrix for 45 degrees clockwise rotation
        angle = torch.tensor(-45*np.pi/180)
        s, c = torch.sin(angle), torch.cos(angle)
        self.rotate = torch.stack([torch.stack([c, -s]),
                                   torch.stack([s, c])]).to(device)        

    def forward(self, X_persis, diagram_slices):
        '''
        X_persis: (num_points, 2) float tensor. Concatenated birth, death pairs of multiple persistence diagrams
        diagram_slices: (num_diag, 2) int tensor. Start and end indices of each persistence diagram. Should be consistent
        with num_points
        
        Returns:
        output: (num_diag, self.num_units) float tensor.
        '''
        X_rot = torch.matmul(self.rotate, X_persis.t()).t()
        mu1 = torch.exp(self.log_mu1) 
        sigma0, sigma1 = torch.exp(self.log_sigma0), torch.exp(self.log_sigma1)
        log_term1 = -(torch.square(sigma0*(X_rot[:, 0].view(-1,1)-self.mu0.view(1,-1))) +
                      torch.square(sigma1*(X_rot[:, 1].view(-1,1)-mu1.view(1,-1))))
        log_term2 = -(torch.square(sigma0*(X_rot[:, 0].view(-1,1)-self.mu0.view(1,-1))) +
                      torch.square(sigma1*((torch.log(X_rot[:, 1]/self.thresh)*self.thresh +
                                           self.thresh).view(-1,1)-mu1.view(1,-1))))
        term1, term2 = torch.exp(log_term1), torch.exp(log_term2)
        cond = (X_rot[:, 0] >= self.thresh).view(-1,1) # Whether to use term1 or term2
        output = cond*term1 + torch.logical_not(cond)*term2
        print(output.shape)
        output_diagwise = [torch.sum(output[start: end], dim = 0) for start, end in diagram_slices]
        output_diagwise = torch.stack(output_diagwise)
        
        return output_diagwise
        
        
if __name__ == '__main__':
    device = torch.device('cuda')
    layer = Topo_Signature_Layer(16)
    layer.cuda()
    persdiag = torch.tensor([[0.2, 0.4], [0.8, 0.8], [1.3, 1.7], [2.4, 3.0], [0.7, 1.5]]).to(device)
    diag_slices = torch.tensor([[0, 3], [3, 4], [4, 5]]).to(device)
    X_out = layer(persdiag, diag_slices)
    print(X_out.shape)
    
        