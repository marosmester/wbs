import numpy as np
import math
import torch
import torch.nn.functional as F
import typing

def hdist(H: torch.Tensor, pts_matches: torch.Tensor):
    '''Function, calculates one-way reprojection error
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error


    Shape:
        - Input :math:`(3, 3)`, :math:`(B, 4)`
        - Output: :math:`(B, 1)`
    '''
    B, _ = pts_matches.shape
    homog_pts = torch.cat( (pts_matches[:, :2], torch.ones( (B, 1) ) ), dim=1 ).unsqueeze(-1)
    transformed_pts = H.unsqueeze(0).repeat(B,1,1) @ homog_pts
    transformed_pts = transformed_pts[:,:2]/transformed_pts[:,2:3]  
    return torch.norm(transformed_pts.squeeze(-1) - pts_matches[:,2:], dim=1, keepdim=True)**2

def sample(pts_matches: torch.Tensor, num: int=4):
    '''Function, which draws random sample from pts_matches
    
    Return:
        torch.Tensor:

    Args:
        pts_matches: torch.Tensor: 2d tensor
        num (int): number of correspondences to sample

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(num, 4)`
    '''
    #sample = torch.zeros(num,4)
    B, _ = pts_matches.shape
    rand_idxs = torch.randperm(B)[:num]
    return pts_matches[rand_idxs]

def check_colinear(min_sample, th):
    """
    Checks if any of the point triplets lie on a line.
    Args:
        min_sample: torch.Tensor: 2d tensor
        th: float 
    """
    B, _ = min_sample.shape
    left_pts = torch.cat( (min_sample[:,:2], torch.ones(B,1)), dim=1  )
    #right_pts = torch.cat( (min_sample[:,2:], torch.ones(B,1)), dim=1  )
    ret = False
    for i in range(4): 
        triple1 = torch.cat((left_pts[:i], left_pts[i+1:]), dim=0)
        #triple2 = torch.cat((right_pts[:i], right_pts[i+1:]), dim=0)
        _, S1, _ = torch.linalg.svd(triple1)
        #_, S2, _ = torch.linalg.svd(triple2)
        rank1 = (S1 > th).sum().item()
        #rank2 = (S2 > th).sum().item()
        if rank1 <= 1:# or rank2 <=1:
            ret = True
            break
    return ret
    
def getH(min_sample):
    '''Function, which estimates homography from minimal sample
    Return:
        torch.Tensor:

    Args:
        min_sample: torch.Tensor: 2d tensor

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(3, 3)`
    '''
    H_norm = torch.eye(3)

    if check_colinear(min_sample, 1e-6):
        H_norm = None
    else:
        # create C matrix
        C = torch.zeros(8, 9)
        for i in range(4):
            j = 2*i
            C[j,:] = torch.tensor( [-min_sample[i,0], -min_sample[i,1], -1,
                                    0, 0, 0,
                                    min_sample[i,2]*min_sample[i,0], min_sample[i,2]*min_sample[i,1], min_sample[i,2]] )
            C[j+1,:] = torch.tensor( [0,0,0,
                                      -min_sample[i,0], -min_sample[i,1], -1,
                                      min_sample[i,3]*min_sample[i,0], min_sample[i,3]*min_sample[i,1], min_sample[i,3]] )

        # calculate h from Ch = 0
        _, _, Vh = torch.linalg.svd(C)
        h = Vh[-1,:]
        h = h/h[-1]     # normalization from the last element
        H_norm = h.reshape(3,3)

    return  H_norm


def nsamples(n_inl:int , num_tc:int , sample_size:int , conf: float):
    eps = 1e-12
    if n_inl == num_tc:
        return 0
    else:
        return math.log(1-conf)/( math.log(1- (n_inl/num_tc)**sample_size ) + eps)
    


def ransac_h(pts_matches: torch.Tensor, th: float = 4.0, conf: float = 0.99, max_iter:int = 1000):
    '''Function, which robustly estimates homography from noisy correspondences
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error

    Args:
        pts_matches: torch.Tensor: 2d tensor
        th (float): pixel threshold for correspondence to be counted as inlier
        conf (float): confidence
        max_iter (int): maximum iteration, overrides confidence
        
    Shape:
        - Input  :math:`(B, 4)`
        - Output: :math:`(3, 3)`,   :math:`(B, 1)`
    '''
    Hbest = torch.eye(3)
    inl_best_mask = torch.zeros(pts_matches.size(0),1) > 0
    B,_ = pts_matches.shape
    supportBest = 0
    new_max_iter = max_iter

    for i in range(max_iter):
        # step 1: sample random points
        rand_sample = sample(pts_matches)

        # step 2: calculate homography
        H = getH(rand_sample)
        if H == None:
            continue

        # step 3: calculate model support
        inl_mask = hdist(H, pts_matches)  < th
        support = ( inl_mask ).sum()
        if support > supportBest:
            supportBest = support
            inl_best_mask = inl_mask
            Hbest = H
            new_max_iter = nsamples(support, B, 4, conf)
            print(new_max_iter)

        # step 4: check for early termination
        if i >= new_max_iter:
            break
        
    return Hbest, inl_best_mask




