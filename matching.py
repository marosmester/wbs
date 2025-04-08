import numpy as np
import math
import torch
import torch.nn.functional as F
import typing

def match_snn(desc1: torch.Tensor, desc2: torch.Tensor, th: float = 0.8):
    '''Function, which finds nearest neightbors for each vector in desc1,
    which satisfy first to second nearest neighbor distance <= th check
    
    Return:
        torch.Tensor: indexes of matching descriptors in desc1 and desc2
        torch.Tensor: L2 desriptor distance ratio 1st to 2nd nearest neighbor


    Shape:
        - Input :math:`(B1, D)`, :math:`(B2, D)`
        - Output: :math:`(B3, 2)`, :math:`(B3, 1)` where 0 <= B3 <= B1
    '''
    #matches_idxs = torch.arange(0, desc2.size(0)).view(-1, 1).repeat(1, 2)
    #match_dists = torch.zeros(desc2.size(0),1)
    
    Dmat = torch.cdist(desc1, desc2, p=2)   # (B1, B2)
    sorted_dists, sorted_idxs = torch.sort(Dmat, dim=1)
    mask = (sorted_dists[:, 0] / sorted_dists[:,1]) <= th # (B1,) boolean mask
    # extract match ratios:
    match_dists = sorted_dists[mask,0]/sorted_dists[mask,1]
    # extract indeces of matched descriptors:
    desc2_idxs = sorted_idxs[mask, 0]
    desc1_idxs = torch.nonzero(mask, as_tuple=True)[0]
    matches_idxs = torch.stack( (desc1_idxs, desc2_idxs), dim=1 )

    return matches_idxs, match_dists
