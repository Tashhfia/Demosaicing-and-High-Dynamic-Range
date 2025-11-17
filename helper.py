import numpy as np, matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pathlib import Path
import cv2
from PIL import Image

def create_mask(dims):
    """ 
    Create R, G, B masks based on image dims. Assumes RGGB pattern.
    """
    h, w = dims[0], dims[1]
    R = np.zeros((h,w), dtype=np.float32)
    G = np.zeros((h,w), dtype=np.float32)
    B = np.zeros((h,w), dtype=np.float32)

    # green mask
    G[0::2, 1::2] = 1.0 
    G[1::2, 0::2] = 1.0  
    # red mask
    R[0::2, 0::2] = 1.0  
    # blue mask
    B[1::2, 1::2] = 1.0 
    return R, G, B

def demosaic(raw_img, kernel_size = (3,3)):
    """
    Returns a demosaiced RGB image
    """
    data_f = raw_img.astype(np.float32)
    kernel = np.ones(kernel_size, dtype=np.float32)
    print(raw_img.shape)
    r_mask, g_mask, b_mask = create_mask(raw_img.shape)

    R = ndi.convolve(data_f * r_mask, kernel, mode='reflect') / ndi.convolve(r_mask, kernel, mode='reflect')
    G = ndi.convolve(data_f * g_mask, kernel, mode='reflect') / ndi.convolve(g_mask, kernel, mode='reflect')
    B = ndi.convolve(data_f * b_mask, kernel, mode='reflect') / ndi.convolve(b_mask, kernel, mode='reflect')
    
    return np.stack((R, G, B), axis=-1)

def norm_img(img, only_image=False):
    """ Normalize image for visualization """
    a = np.percentile(img, 0.01)
    b = np.percentile(img, 99.99)
    norm_image = (img - a)/ (b-a)
    norm_image[norm_image<0] = 0
    norm_image[norm_image>1] = 1

    if only_image:
        return norm_image.astype(np.float32)

    return norm_image.astype(np.float32), a, b

def denorm(normed, a, b):
    """ Denormalize image from [0,1] to [a,b] """
    return normed * (b - a) + a

def visualize_image(img, title="Image"):
    """ Visualize image"""
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def lecture_gamma(normed_img, gamma = 0.3):
    """
    y = x^gamma where higher gamma = darker image
    """
    return normed_img ** gamma

def gray_world(image):
    """ Applies gray world for white balancing"""
    i_mean = np.mean(image)
    red_mean = np.mean(image[:,:,0])
    green_mean = np.mean(image[:,:,1])
    blue_mean = np.mean(image[:,:,2])

    scaled_r = image[:,:,0] * i_mean / red_mean
    scaled_g = image[:,:,1] * i_mean / green_mean
    scaled_b = image[:,:,2] * i_mean / blue_mean

    return np.stack((scaled_r, scaled_g, scaled_b), axis=-1)

def HDR(data):
    h_img = data[0][0]
    h_img = h_img.astype(np.float32)
    max_exp = data[0][1]    # first image has max exposure
    
    # iterate over the rest of the images
    for i_img, curr_exp in data[1:]:
        i_img = i_img.astype(np.float32)
        exp_change = max_exp / curr_exp

        i_scaled = i_img * exp_change
        thresh = 0.8 * np.max(h_img)

        # replace pixels in h that are too birght with the ones in 
        # scaled i
        mask = h_img >= thresh
        h_img[mask] = i_scaled[mask]
        
    return h_img.astype(np.float32)

def log_norm(image):
    """ Apply logarithmic normalization to the image """
    # Apply logarithm (add 1 to avoid log(0))
    im_log = np.log(image + 1)
    log_min = np.min(im_log)
    log_max = np.max(im_log)
    # Normalize to [0, 255] range
    im_log_norm = (im_log - log_min) / (log_max - log_min) * 255

    return im_log_norm.astype(np.uint8)
