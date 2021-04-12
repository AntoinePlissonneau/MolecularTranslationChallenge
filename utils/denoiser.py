# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:52:52 2021

@author: aplissonneau
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import os

def noise_remover(n, img):
    """ 
    Remove the noise of an input image
    """
    conv = nn.Conv2d(1, 1, (n,n), padding=int((n-1)/2), bias=False)
    conv.weight = torch.nn.Parameter(torch.ones(n, n)[None, None, :, :])
    conv.require_grad = False
    denoised = img.reshape(-1).copy()
    denoised[(conv(torch.Tensor(img[None, None, :, :])).detach().numpy() <= 255).squeeze().reshape(-1)] = 0
    img = denoised.reshape(img.shape)
    return img

def add_diff(origin_img, processed_img):
    """ 
    Add the diff between origin img and processed img to processed imgin red
    """
    
    diff =  (processed_img - origin_img)
    
    #dilation for improved visualization of removed elements
    kernel = np.ones((3,3),np.uint8)
    diff = cv2.dilate(diff,kernel,iterations = 1)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    processed_img[:,:,1] = processed_img[:,:,1] - diff
    processed_img[:,:,0] = processed_img[:,:,0] - diff

    return processed_img

def process_img(in_path=None, out_path=None, n=7, diff=False, mode="save"):
    """
    Remove the noise of an input image, given its path
    
    in_path: path of the input image
    out_path: path to save the processed image (used if mode="save")
    n: size of the conv kernel (nxn)
    diff: If True, add to the processed image the removed noise in red (for visualisation)
    mode: "save" for save the image, "show" for only plot
    """
    # Read inverse img
    img = 255 - cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    
    # Remove noise
    img2 = noise_remover(n,img)    
    
    # Inverse images to normal
    img = 255 - img
    img2 = 255 - img2
    
    if diff:
        img2 = add_diff(img, img2)
        
    if mode == "save":
        cv2.imwrite(out_path, img2)
        
    if mode == "show":
        cv2.imshow(f"Denoizer n={n}", img2)
    return img2

def process_imgs(folder_path=None, out_folder_path=None, n=7, diff = False):
    """
    Apply process_img for all images in input folder
    
    folder_path: folder where the input imgs are readed
    out_folder_path: folder where the processed images are written
    n: size of the conv kernel (nxn)
    diff: If True, add to the processed image the removed noise in red (for visualisation)
    """
    for file in os.listdir(folder_path):
        in_path = os.path.join(folder_path,file)
        out_path = os.path.join(out_folder_path,file)
        process_img(in_path, out_path, n, diff, "save")
    



if __name__ == "__main__":
    # =============================================================================
    #     Show results for one image
    # =============================================================================
    filename = "111a368fe762.png"
    in_path = os.path.join("bms-molecular-translation/train/1/1/1", filename)
    process_img(in_path, diff = True, mode = "show")
    
    # =============================================================================
    #     Process and save all imgs of a folder for differents kernel sizes
    # =============================================================================
    folder_path = "bms-molecular-translation/train/1/1/1"
    
    for n in [5,7,9]:
        out_folder_path = f"processed_test/{n}"
        os.makedirs(out_folder_path, exist_ok = True) 
        process_imgs(folder_path, out_folder_path, n, diff= True)
    


