import numpy as np
import math
import torch
import torch.nn.functional as F
import typing


def get_gausskernel_size(sigma, force_odd = True):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    if ksize % 2  == 0 and force_odd:
        ksize +=1
    return int(ksize)


def gaussian1d(x: torch.Tensor, sigma: float) -> torch.Tensor: 
    '''Function that computes values of a (1D) Gaussian with zero mean and variance sigma^2'''
    #out =  torch.zeros(x.shape)
    out = (1/ (np.sqrt(2*np.pi) * sigma)) * torch.exp(-(x*x)/(2*sigma**2))
    return out


def gaussian_deriv1d(x: torch.Tensor, sigma: float) -> torch.Tensor:  
    '''Function that computes values of a (1D) Gaussian derivative'''
    out =  -(x/(sigma**2))*gaussian1d(x, sigma)
    return out

def filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    ## Do not forget about flipping the kernel!
    ## See in details here https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5
    
    # expand the kernel correctly:
    c = x.shape[1]
    if len(kernel.shape) == 2:
        kernel = kernel.expand(c, 1, kernel.shape[0], kernel.shape[1])
    # pad the input beforehand:
    pad_height = kernel.shape[2] //2
    pad_width =  kernel.shape[3] //2
    pad2d = (pad_width, pad_width, pad_height, pad_height)
    x = F.pad(x, pad2d, mode = 'replicate')
    # rotate kernel:
    kernel = torch.flip(kernel, dims = [0, 1])
    # convolve:
    out =  F.conv2d(x, kernel, groups=c)
    return out

def gaussian_filter2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.

    Arguments:
        sigma (Tuple[float, float]): the standard deviation of the kernel.
        
    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    """
    c = x.shape[1] 
    # prepare 1d gaussian kernel
    ksize = get_gausskernel_size(sigma)
    kx = torch.arange(-ksize//2+1, ksize//2+1)
    kernel = gaussian1d(kx, sigma)
    
    # height
    kernel = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(-1)          #kernel = kernel.unsqueeze(0).unsqueeze(0)
    vertical_kernel = kernel.expand((c,1,ksize,1))             #vertical_kernel = kernel.unsqueeze(-1)
    # width
    horiz_kernel = vertical_kernel.reshape((c, 1, 1, ksize))
    
    # perform 2 convolutions:
    out = filter2d(x, vertical_kernel)
    out = filter2d(out, horiz_kernel)
    return out

def spatial_gradient_first_order(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """
    b, c, h, w = x.shape
    ksize = get_gausskernel_size(sigma)
    out =  torch.zeros(b,c,2,h,w)
    
    kx = torch.arange(-ksize//2+1, ksize//2+1)
    gauss_kernel = gaussian1d(kx, sigma)
    gauss_deriv_kernel = gaussian_deriv1d(kx, sigma)
    out_dx = filter2d(x, gauss_kernel.unsqueeze(-1))
    out_dx = filter2d(out_dx, -gauss_deriv_kernel.unsqueeze(0) ) 
    
    out_dy = filter2d(x, gauss_kernel.unsqueeze(0))
    out_dy = filter2d(out_dy, -gauss_deriv_kernel.unsqueeze(-1) )

    out[:,:,0,:,:] = out_dx
    out[:,:,1,:,:] = out_dy
    
    # ensure exact zero if necessary
    out[out.abs() < 1e-8] = 0
    return out

def affine(center: torch.Tensor, unitx: torch.Tensor, unity: torch.Tensor) -> torch.Tensor:
    r"""Computes transformation matrix A which transforms point in homogeneous coordinates from canonical coordinate system into image

    Return:
        torch.Tensor: affine tranformation matrix

    Shape:
        - Input :math:`(B, 2)`, :math:`(B, 2)`, :math:`(B, 2)` 
        - Output: :math:`(B, 3, 3)`

    """
    assert center.size(0) == unitx.size(0)
    assert center.size(0) == unity.size(0)
    B = center.size(0)
    
    # construct X matrix
    to_append = torch.ones((B, 1))
    center = torch.cat( (center, to_append), 1 )
    unitx = torch.cat( (unitx, to_append) , 1)
    unity = torch.cat( (unity, to_append), 1) 
    X = torch.stack([center, unitx, unity], dim = 2)

    # construct Y matrix of canonical base
    Y = torch.tensor([[0, 1, 0],        # canon unit vectors
                      [0, 0, 1],
                      [1, 1, 1]]).to(torch.float32)
    Y = Y.unsqueeze(0).repeat(B,1,1)

    # compute transformation A
    A = X @ torch.linalg.inv(Y)  
    return A

def extract_affine_patches(input: torch.Tensor,
                           A: torch.Tensor,
                           img_idxs: torch.Tensor,
                           PS: int = 32,
                           ext: float = 6.0):
    """Extract patches defined by affine transformations A from image tensor X.
    
    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors. 

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    b,ch,h,w = input.size()
    num_patches = A.size(0)
    # grid creation:
    grid_y, grid_x = torch.meshgrid(torch.linspace(-ext, ext, PS),
                                    torch.linspace(-ext, ext, PS),
                                    indexing="ij" )    
    grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim = 0 )
    grid = grid.reshape(3,-1)      # should reshape it into (3, PS*PS)

    # grid affine transformation 
    transformed_grid = A[:,:2,:] @ grid
    transformed_grid = transformed_grid.permute(0, 2, 1).reshape( num_patches, PS, PS, 2) # (N, PS, PS, 2)
    # normalization:
    transformed_grid[:,:,:, 0] = 2 * (transformed_grid[:,:,:, 0]/w ) - 1
    transformed_grid[:,:,:, 1] = 2 * (transformed_grid[:,:,:, 1]/h ) - 1

    # slice the input based on number of patches
    input = input[img_idxs.squeeze(-1)]

    #extract patches
    patches = F.grid_sample(input, transformed_grid, mode="bilinear")
    return patches # (N, CH, PS, PS)


def extract_antializased_affine_patches(input: torch.Tensor,
                           A: torch.Tensor,
                           img_idxs: torch.Tensor,
                           PS: int = 32,
                           ext: float = 6.0):
    """Extract patches defined by affine transformations A from scale pyramid created image tensor X.
    It runs your implementation of the `extract_affine_patches` function, so it would not work w/o it.
    You do not need to ever modify this finction, implement `extract_affine_patches` instead.
    
    Args:
        input: (torch.Tensor) images, :math:`(B, CH, H, W)`
        A: (torch.Tensor). :math:`(N, 3, 3)`
        img_idxs: (torch.Tensor). :math:`(N, 1)` indexes of image in batch, where patch belongs to
        PS: (int) output patch size in pixels, default = 32
        ext (float): output patch size in unit vectors. 

    Returns:
        patches: (torch.Tensor) :math:`(N, CH, PS,PS)`
    """
    import kornia
    b,ch,h,w = input.size()
    num_patches = A.size(0)
    scale = (kornia.feature.get_laf_scale(ext * A.unsqueeze(0)[:,:,:2,:]) / float(PS))[0]
    half: float = 0.5
    pyr_idx = (scale.log2()).relu().long()
    cur_img = input
    cur_pyr_level = 0
    out = torch.zeros(num_patches, ch, PS, PS).to(device=A.device, dtype=A.dtype)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        _, ch_cur, h_cur, w_cur = cur_img.size()
        scale_mask = (pyr_idx == cur_pyr_level).squeeze()
        if (scale_mask.float().sum()) >= 0:
            scale_mask = (scale_mask > 0).view(-1)
            current_A = A[scale_mask]
            current_A[:, :2, :3] *= (float(h_cur)/float(h))
            patches = extract_affine_patches(cur_img,
                                 current_A, 
                                 img_idxs[scale_mask],
                                 PS, ext)
            out.masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.geometry.pyrdown(cur_img)
        cur_pyr_level += 1
    return out
