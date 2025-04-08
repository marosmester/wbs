import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
import kornia                       # added
import matplotlib.pyplot as plt     # added
from imagefiltering import * 

def imshow_torch_channels(tensor, dim = 1, *kwargs):
    num_ch = tensor.size(dim)
    fig=plt.figure(figsize=(num_ch*5,5))
    tensor_splitted = torch.split(tensor, 1, dim=dim)
    for i in range(num_ch):
        fig.add_subplot(1, num_ch, i+1)
        plt.imshow(kornia.tensor_to_image(tensor_splitted[i].squeeze(dim)), *kwargs)
    return


def harris_response(x: torch.Tensor,
                     sigma_d: float,
                     sigma_i: float,
                     alpha: float = 0.04)-> torch.Tensor:
    r"""Computes the Harris cornerness function.The response map is computed according the following formulation:

    .. math::
        R = det(M) - alpha \cdot trace(M)^2

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): sigma of Gaussian derivative
        sigma_i (float): sigma of Gaussian blur, aka integration scale
        alpha (float): constant

    Return:
        torch.Tensor: Harris response

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    b, c, h, w = x.shape
    gradient_img = spatial_gradient_first_order(x, sigma_d)
    Dx = gradient_img[:,:,0,:,:]
    Dy = gradient_img[:,:,1,:,:]
    Dxy_smooth = gaussian_filter2d(Dx*Dy, sigma_i)
    Dx_smooth = gaussian_filter2d(Dx*Dx, sigma_i)
    Dy_smooth = gaussian_filter2d(Dy*Dy, sigma_i)
    det = Dx_smooth * Dy_smooth - Dxy_smooth*Dxy_smooth
    trace = Dx_smooth + Dy_smooth
    return det - alpha*trace**2

def nms2d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the feature map in 3x3 neighborhood.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
    Return:
        torch.Tensor: nmsed input

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    b, c, h, w = x.shape
    x_pad = F.pad(x, (1,1,1,1), mode = 'constant', value= 0)
    x_unfold = F.unfold(x_pad, kernel_size=3)               # shape: (B, C * 9, H * W) ... unfold effectively removes the padding
    x_unfold = x_unfold.reshape(b, c, 9, h, w)
    x_unfold_zeroed = x_unfold.clone()                  # make a copy
    x_unfold_zeroed[:, :, 4, :, :] = 0                  # zero out the center elements
    max_vals, _ = torch.max(x_unfold_zeroed, dim=2)     # shape: (B, C, H, W) 
    mask = (x > max_vals) & (x > th)
    return x*mask


def harris(x: torch.Tensor, sigma_d: float, sigma_i: float, th: float = 0):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): scale
        sigma_i (float): scale
        th (float): threshold

    Return:
        torch.Tensor: coordinates of local maxima in format (b,c,h,w)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 4)`, where N - total number of maxima and 4 is (b,c,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    hr = harris_response(x, sigma_d, sigma_i)
    hr_supressed = nms2d(hr, th)
    ret = torch.nonzero(hr_supressed)
    #print(f"ret.hsape={ret.shape}")
    return ret


def create_scalespace(x: torch.Tensor, n_levels: int, sigma_step: float):
    r"""Creates a scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur.
    Args:
        x: torch.Tensor :math:`(B, C, H, W)`
        n_levels (int): number of the levels.
        sigma_step (float): blur step.

    Returns:
        Tuple(torch.Tensor, List(float)):
        1st output: image pyramid, (B, C, n_levels, H, W)
        2nd output: sigmas (coefficients for scale conversion)
    """
    b, ch, h, w = x.size()
    out = torch.zeros(b, ch, n_levels, h, w)
    out[:,:,0,:,:] = x
    sigma_i = 1
    sigmas  = [sigma_i for i in range(n_levels)]
    for i in range(1, n_levels):
        sigma_i = sigma_step * sigma_i
        sigmas[i] = sigma_i
        out[:,:,i,:,:] = gaussian_filter2d(x, sigma_i)
    #out = torch.zeros(b, ch, n_levels, h, w), [1.0 for x in range(n_levels)]
    return out, sigmas

def nms3d(x: torch.Tensor, th: float = 0):
    """
    Applies strict non-maximum suppression to a 5D tensor in a 3x3x3 neighborhood.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        th (float): Threshold, values below this are suppressed.

    Returns:
        torch.Tensor: Suppressed tensor of shape (B, C, D, H, W).
    """
    b, c, d, h, w = x.shape
    #print(f" nms3d: x max = {torch.max(x)}")  
    x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='constant', value=0)  
    # extract all 3×3×3 neighborhoods
    x_unfold_d = x_padded.unfold(2, 3, 1)  
    x_unfold_h = x_unfold_d.unfold(3, 3, 1)  
    x_unfold_w = x_unfold_h.unfold(4, 3, 1)  

    x_patches = x_unfold_w.contiguous().view(b, c, d, h, w, 27)
    x_patches[:, :, :, :, :, 13] = 0  # idx 13 is the center of a 3x3x3 patch

    max_pooled, _ = torch.max(x_patches, dim=-1)  # shape: (b, c, d, h, w)
    mask = (x > max_pooled) & (x > th)
    return x*mask

def scalespace_harris_response(x: torch.Tensor,
                                n_levels: int = 40,
                                sigma_step: float = 1.1):
    r"""First computes scale space and then computes the Harris cornerness function 
    Args:
        x: torch.Tensor: 4d tensor
        n_levels (int): number of the levels, (default 40)
        sigma_step (float): blur step, (default 1.1)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, N_LEVELS, H, W)`, List(floats)
    """
    b, c, h, w = x.shape
    ss, sigmas = create_scalespace(x, n_levels, sigma_step)
    sshr = torch.zeros_like(ss) #torch.zeros_like(x).unsqueeze(2).expand(b, c, n_levels, h, w)
    sigma_d = 1.15
    sigma_i = 1.2
    for i in range(n_levels):
        sshr[:,:,i,:,:] = (sigmas[i]**4) * harris_response(ss[:,:,i,:,:], sigma_d, sigma_i)
    return sshr, sigmas


def scalespace_harris(x: torch.Tensor,
                       th: float = 0,
                       n_levels: int = 40,
                       sigma_step: float = 1.1):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
        n_levels (int): number of scale space levels (default 40)
        sigma_step (float): blur step, (default 1.1)
        
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 5)`, where N - total number of maxima and 5 is (b,c,d,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    # Don't forget to convert scale index to scale value with use of sigma
    sshr, sigmas = scalespace_harris_response(x, n_levels, sigma_step)
    sigmas = torch.tensor(sigmas)
    nmsed = nms3d(sshr, th)
    locations = torch.nonzero(nmsed)
    sigma_idxs = locations[:,2]
    locations[:,2] = sigmas[sigma_idxs]
    return locations

