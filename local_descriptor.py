import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from typing import Tuple
from imagefiltering import * 
from local_detector import *


def affine_from_location(b_ch_d_y_x: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)` 
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    # matrix A:
    N = b_ch_d_y_x.shape[0]
    yx = b_ch_d_y_x[:, -2:]
    xy = (yx[:, [1,0]] ).unsqueeze(-1)
    scale = b_ch_d_y_x[:, 2]  
    scale = scale.unsqueeze(-1).unsqueeze(-1).repeat(1,2,2)
    scale[:,0,1] = 0
    scale[:,1,0] = 0
    A = torch.cat((scale, xy), dim=2)
    homog = (torch.tensor([0,0,1]) ).unsqueeze(0).unsqueeze(-1).repeat(N,1,1).reshape(N,1,3)
    A = torch.cat( (A, homog), dim=1 )
    
    # image indexes:
    img_idxs = b_ch_d_y_x[:,0].long() # torch.arange(0, N).unsqueeze(-1)
    return A.float(), img_idxs


def affine_from_location_and_orientation(b_ch_d_y_x: torch.Tensor,
                                         ori: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian). Ori - orientation angle in radians
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`, :math:`(B, 1) 
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    #A = torch.zeros(b_ch_d_y_x.size(0), 3, 3)
    #img_idxs = torch.zeros(b_ch_d_y_x.size(0), 1).long()
    N = b_ch_d_y_x.shape[0]
    R = torch.stack(
        [torch.cos(ori), torch.sin(ori), torch.zeros_like(ori),
        -torch.sin(ori), torch.cos(ori), torch.zeros_like(ori),
        torch.zeros_like(ori), torch.zeros_like(ori), torch.ones_like(ori)],
         dim=1).reshape(N, 3, 3)
    A, _ = affine_from_location(b_ch_d_y_x)
    T = A.float() @ R
    
    # image indexes:
    img_idxs = b_ch_d_y_x[:,0].long() 
    return T, img_idxs


def affine_from_location_and_orientation_and_affshape(b_ch_d_y_x: torch.Tensor,
                                                      ori: torch.Tensor,
                                                      aff_shape: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image
    from keypoint location (output of scalespace_harris or scalespace_hessian)
    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 5)`, :math:`(B, 1), :math:`(B, 3)
        - Output: :math:`(B, 3, 3)`, :math:`(B, 1)`

    """
    # xy 
    N = b_ch_d_y_x.shape[0]
    yx = b_ch_d_y_x[:, -2:]
    xy = (yx[:, [1,0]] ).unsqueeze(-1)

    # A1
    C = torch.stack( (aff_shape[:, :2], aff_shape[:,1:]), dim=1)
    sigma = b_ch_d_y_x[:,2].unsqueeze(-1).unsqueeze(-1)
    sqrtC_inv = torch.linalg.cholesky( torch.linalg.inv(C) )    # cholesky inverse
    A1 =  sigma * torch.linalg.inv( sqrtC_inv / ( torch.sqrt(torch.linalg.det(sqrtC_inv)).unsqueeze(-1).unsqueeze(-1) ))
    
    # A_norm
    A_norm = torch.cat((A1, xy), dim=2)
    homog = (torch.tensor([0,0,1]) ).unsqueeze(0).unsqueeze(-1).repeat(N,1,1).reshape(N,1,3)
    A_norm = torch.cat( (A_norm, homog), dim=1 )
    
    # R
    R = torch.stack(
    [torch.cos(ori), torch.sin(ori), torch.zeros_like(ori),
    -torch.sin(ori), torch.cos(ori), torch.zeros_like(ori),
    torch.zeros_like(ori), torch.zeros_like(ori), torch.ones_like(ori)],
        dim=1).reshape(N, 3, 3)
    
    # A
    A = A_norm @ R
    
    # image indexes:
    img_idxs = b_ch_d_y_x[:,0].long() 
    return A, img_idxs


def estimate_patch_dominant_orientation(x: torch.Tensor, num_angular_bins: int = 36):
    """Function, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.
    
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
        num_angular_bins: int, default is 36
    
    Returns:
        angles: (torch.Tensor) in radians shape [Bx1]
    """
    b,_,_,ps = x.shape

    # magnitude and orientation of gradients:
    grad = spatial_gradient_first_order(x, sigma = ps/17)
    Dx = grad[:,:,0,:,:]
    Dy = grad[:,:,1,:,:]
    mag = torch.sqrt(Dx**2 + Dy**2)
    orient = torch.atan2(Dy, Dx)
    orient[orient < 0] += 2*np.pi   # transform from [-pi, pi) to [0, 2pi)
    
    # 2d gaussian kernel (PS, PS)
    sigma_i = ps/5
    kx = torch.arange(-ps//2+1, ps//2+1)
    gauss1d = gaussian1d(kx, sigma_i)
    gauss2d = torch.outer(gauss1d, gauss1d)
    gauss2d = gauss2d / gauss2d.sum()

    # histogram
    binsize = 2*np.pi/num_angular_bins
    angles = torch.zeros(b)
    for i in range(b):
        hist,_ = torch.histogram(orient[i,0,:,:], 
                            bins= num_angular_bins,
                            range= (0., 2*np.pi),
                            weight = mag[i,0,:,:] * gauss2d)        # element-wise multiplication with gauss2d
        max_bin_idx = torch.argmax(hist)
        angles[i] = binsize/2 + max_bin_idx*binsize      # /2 because middle of the bin?
    return angles

def estimate_patch_affine_shape(x: torch.Tensor):
    """Function, which estimates the patch affine shape by second moment matrix. Returns ellipse parameters: a, b, c
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
    
    Returns:
        ell: (torch.Tensor) in radians shape [Bx3]
    """
    sigma_i = 1.2
    sigma_d = 1.3
    grad = spatial_gradient_first_order(x, sigma = sigma_d)
    Dx = grad[:,:,0,:,:]
    Dy = grad[:,:,1,:,:]
    Dxy = gaussian_filter2d(Dx*Dy, sigma_i)
    Dx2 = gaussian_filter2d(Dx*Dx, sigma_i)
    Dy2 = gaussian_filter2d(Dy*Dy, sigma_i)
    c11 = Dx2.sum(dim = (2,3))
    c12 = Dxy.sum(dim = (2,3))
    c22 = Dy2.sum(dim = (2,3))
    out = torch.cat( (c11,  torch.cat( (c12, c22), dim = 1) ), dim =1)
    return out


def pick_subpatch(input, miniPS, num_spatial_bins, idx):
    """
    row-major order
    """
    row = idx//num_spatial_bins
    col = idx%num_spatial_bins
    return input[:, :, row*miniPS:(row+1)*miniPS , col*miniPS:(col+1)*miniPS]

def calc_sift_descriptor(input: torch.Tensor,
                  num_ang_bins: int = 8,
                  num_spatial_bins: int = 4,
                  clipval: float = 0.2) -> torch.Tensor:
    '''    
    Args:
        x: torch.Tensor (B, 1, PS, PS)
        num_ang_bins: (int) Number of angular bins. (8 is default)
        num_spatial_bins: (int) Number of spatial bins (4 is default)
        clipval: (float) default 0.2
        
    Returns:
        Tensor: SIFT descriptor of the patches

    Shape:
        - Input: (B, 1, PS, PS)
        - Output: (B, num_ang_bins * num_spatial_bins ** 2)
    '''
    #out = torch.zeros(input.size(0), num_ang_bins * num_spatial_bins ** 2)
    B,_,_,PS = input.shape
    miniPS = PS // num_spatial_bins

    # photometric norm
    input = photonorm(input)

    # magnitude and orientation of gradients:
    grad = spatial_gradient_first_order(input, sigma = 1.6)
    Dx = grad[:,:,0,:,:]
    Dy = grad[:,:,1,:,:]
    mag = torch.sqrt(Dx**2 + Dy**2)
    ori = torch.atan2(Dy, Dx)
    ori[ori < 0] += 2*np.pi   # transform from [-pi, pi) to [0, 2pi)

    # 2d gaussian kernel (PS, PS) weight on magnitude
    sigma_i = PS/2
    kx = torch.arange(-PS//2+1, PS//2+1)
    gauss1d = gaussian1d(kx, sigma_i)
    gauss2d = torch.outer(gauss1d, gauss1d)
    gauss2d = gauss2d / gauss2d.sum()
    mag = mag * gauss2d     # element-wise multiplication with gaussian window

    # create histogram for each bin and batch
    out = torch.zeros(B, num_spatial_bins ** 2, num_ang_bins)
    for i in range(num_spatial_bins**2):        # spatial bin loop
        batch_hist = torch.zeros(B, num_ang_bins)
        subpatch_ori = pick_subpatch(ori, miniPS, num_spatial_bins, i)
        subpatch_mag = pick_subpatch(mag, miniPS, num_spatial_bins, i)
        for j in range(B):                      # batch loop
            batch_hist[j,:],_ = torch.histogram(subpatch_ori[j, 0, :, :], 
                            bins= num_ang_bins,
                            range= (0., 2*np.pi),
                            weight = subpatch_mag[j,0,:,:])
        
        out[:, i, : ] = batch_hist

    out = out.transpose(-2, -1)
    out = out.reshape(B, num_ang_bins* num_spatial_bins**2)         
    out = torch.clip(out, max=clipval) 
    out /= (out.norm(dim=1, keepdim=True) + 1e-12)  # L2 normalization  
    return out

def photonorm(x: torch.Tensor):
    """Function, which normalizes the patches such that the mean intensity value per channel will be 0 and the standard deviation will be 1.0. Values outside the range < -3,3> will be set to -3 or 3 respectively
    Args:
        x: (torch.Tensor) shape [BxCHxHxW]
    
    Returns:
        out: (torch.Tensor) shape [BxCHxHxW]
    """
    mean = x.mean( dim= (2, 3), keepdim=True)
    std = x.std( dim=(2,3), keepdim = True)
    normalized = (x - mean)/(std + 1e-6)        # prevent zero division error
    out = normalized.clamp(min= -3.0, max= 3.0)
    return out



