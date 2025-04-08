import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Tuple
import kornia

from imagefiltering import *
from local_descriptor import *
from local_detector import *
from matching import *
from ransac import *

def get_MAE_imgcorners(h,w, H_gt, H_est):
    '''Example of usage:
    H_gt = np.loadtxt(Hgt)
    img1 = K.image_to_tensor(cv2.imread(f1,0),False)/255.
    img2 = K.image_to_tensor(cv2.imread(f2,0),False)/255.
    h = img1.size(2)
    w = img1.size(3)
    H_out = matchImages(img1,img2)
    MAE = get_MAE_imgcorners(h,w,H_gt, H_out.detach().cpu().numpy())
    '''
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H_est).squeeze(1)
    dst_GT = cv2.perspectiveTransform(pts, H_gt).squeeze(1)
    error = np.abs(dst - dst_GT).sum(axis=1).mean()
    return error

# After per-pair MAE calculation, it is reduced as following:
# acc = []
# for th in [0.1, 1.0, 2.0, 5.0, 10., 15.]:
#    A = (np.array(MAEs) <= th).astype(np.float32).mean()
#    acc.append(A)
# MAA = np.array(acc).mean()
# print (MAA)

#---------------------------------------------------------------------------------------

# loading:

def timg_load(fname, to_gray = True):
    img = cv2.imread(fname)
    with torch.no_grad():
        timg = kornia.image_to_tensor(img, False).float()
        if to_gray:
            timg = kornia.color.bgr_to_grayscale(timg)
        else:
            timg = kornia.color.bgr_to_rgb(timg)
    return timg

def imshow_torch(tensor,figsize=(8,6), *kwargs):
    plt.figure(figsize=figsize)
    plt.imshow(kornia.tensor_to_image(tensor), *kwargs)
    return

# connecting the pipepline together:

def detect_and_describe(img,
                        det='harris',
                        th=0.00001,
                        affine=False,
                        PS = 31):
    if det.lower() == 'harris':
        keypoint_locations = scalespace_harris(img, th, 20, 1.3)
    else:
        raise ValueError('Unknown detector, try harris')
    n_kp = keypoint_locations.size(0)
    A, img_idxs = affine_from_location(keypoint_locations)
    if affine:
        patches  = extract_affine_patches(img, A, img_idxs, 19, 5.0)
        aff_shape = estimate_patch_affine_shape(patches)
        dummy_angles = torch.zeros(n_kp,1, dtype=torch.float, device=img.device)
                                                          
        A, img_idxs = affine_from_location_and_orientation_and_affshape(keypoint_locations, 
                                                          dummy_angles,
                                                          aff_shape)
    patches =  extract_affine_patches(img, A, img_idxs, 19, 5.0)
    ori = estimate_patch_dominant_orientation(patches)
    if affine:
        A, img_idxs = affine_from_location_and_orientation_and_affshape(keypoint_locations, 
                                                  ori,
                                                  aff_shape)
    else:
        A, img_idxs = affine_from_location_and_orientation(keypoint_locations, 
                                                  ori)
    patches =  extract_affine_patches(img, A, img_idxs, PS, 10.0)
    descs = calc_sift_descriptor(patches)
    return keypoint_locations, descs, A

def matchImages(timg1: torch.Tensor,
                timg2: torch.Tensor):
    r"""Returns the homography, which maps image 1 into image 2
    Args:
        timg1: torch.Tensor: 4d tensor of shape [1x1xHxW]
        timg2: torch.Tensor: 4d tensor of shape [1x1xH2xW2]
    Returns:
        H: torch.Tensor: [3x3] homography matrix
        
    Shape:
      - Input: :math:`(1, 1, H, W)`, :math:`(1, 1, H, W)`
      - Output: :math:`(3, 3)`
    """
    keypts1, desc1, A1 = detect_and_describe(timg1, 'harris', 0.00001, False)
    keypts2, desc2, A2 = detect_and_describe(timg2, 'harris', 0.00001, False)
    keypts1 = keypts1[:,3:]
    keypts2 = keypts2[:,3:]
    match_idxs, match_dists = match_snn(desc1, desc2) 
    pts_matches = torch.cat( (keypts1[match_idxs[:,0]], keypts2[match_idxs[:,1]]), dim=1)
    H_best, _ = ransac_h(pts_matches, max_iter=1000)
    return H_best

if __name__ == "__main__":
    imgPath1 = "bikes/img1.ppm"
    imgPath2 = "bikes/img2.ppm"
    HgtPath = "bikes/H1to2p"    

    H_gt = np.loadtxt(HgtPath)
    img1 = kornia.image_to_tensor(cv2.imread(imgPath1,0),False)/255.
    img2 = kornia.image_to_tensor(cv2.imread(imgPath2,0),False)/255.
    h = img1.size(2)
    w = img1.size(3)
    H_out = matchImages(img1,img2)
    MAE = get_MAE_imgcorners(h,w,H_gt, H_out.detach().cpu().numpy())
    print(H_out, MAE)