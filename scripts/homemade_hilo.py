import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import tifffile as tif
import glob


##################################    useful functions   #####################################


def create_patterned_image(image_path, output_path, square_size=10, inverted=False):
    """ Loads an image and superposes a checkerboard on it."""
    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    # Create a mask for the checkerboard pattern
    if inverted==True:
        mask = Image.new("L",(width, height),0) 
    else:
        mask = Image.new("L",(width, height),255)
    draw = ImageDraw.Draw(mask)
    
    for y in range(0, height, square_size*2):
        for x in range(0, width, square_size*2):
            if inverted==True:
                draw.rectangle([x, y, x+square_size-1, y+square_size-1], fill=255)  # Black square on dmd
                draw.rectangle([x + square_size, y+square_size, x+2*square_size-1, y+2*square_size-1], fill=255)
            else:
                draw.rectangle([x, y, x+square_size-1, y+square_size-1], fill=0)  # White square on dmd
                draw.rectangle([x+square_size, y+square_size, x+2*square_size-1, y+2*square_size-1], fill=0)
    
    # Apply mask to image
    img_with_checkerboard = Image.composite(Image.new("RGB", (width, height), "black"), img, mask)
    
    # Save as tiff
    img_with_checkerboard.save(output_path, format='TIFF')
    print(f"Image with checkerboard pattern saved to {output_path}")



def compute_local_contrast(image, window_size=200):
    """Compute local contrast using local standard deviation."""
    kernel = np.ones((window_size, window_size), np.float32) / (window_size**2) # Create moving window
    
    local_mean = cv2.filter2D(image, -1, kernel) # Compute local mean
    local_mean_sq = cv2.filter2D(image ** 2, -1, kernel) # Compute local squared mean

    # Compute local variance (E[X^2] - (E[X])^2) and take sqrt for standard deviation
    local_stdev = np.sqrt(local_mean_sq - local_mean ** 2)
    return local_stdev





############################################  filters  ##########################################


def low_pass_filter(shape, radius):
    """Create an ideal low-pass filter mask."""
    rows, cols = shape
    center = (rows // 2, cols // 2)
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center[0])**2 + (j - center[1])**2) <= radius:
                mask[i, j] = 1
    return mask


def high_pass_filter(shape, radius):
    """Create an ideal high-pass filter mask."""
    return 1 - low_pass_filter(shape, radius)


#######################################  hilo reconstructions   ##################################


def hilo_reconstruction_contrasted(I_high, I_1, I_2, alpha=1.0, sigma=5, contrast_threshold=0.5):
    """ HiLo image reconstruction.
    Parameters:
    - I_high: uniform illumination.
    - I_1: checkerboard illumination.
    - I_2: inverted checkerboard illumination.
    - alpha: high-frequency contribution.
    - sigma: stdev for Gaussian filtering.
    - contrast_threshold: (NOT USED)
    Returns:
    - I_hilo: reconstructed HiLo image."""
    
    I_high = I_high.astype(np.float32)
    I_1 = I_1.astype(np.float32)
    I_2 = I_2.astype(np.float32)

    # Compute local contrast of checkerboard images
    local_contrast1 = compute_local_contrast(I_1)
    local_contrast2 = compute_local_contrast(I_2)
    # Normalize local contrast
    contrast_weight1 = cv2.normalize(local_contrast1, None, 0, 1, cv2.NORM_MINMAX)
    contrast_weight2 = cv2.normalize(local_contrast2, None, 0, 1, cv2.NORM_MINMAX)

    # Create a mask for in-focus regions based on local contrast (NOT USED)
    # mask1 = np.abs(local_contrast1) > contrast_threshold
    # mask2 = np.abs(local_contrast2) > contrast_threshold

    # Compute the low-frequency component as the average of checkerboard images (NOT USED)
    # I_low = (I_1 * mask1.astype(np.float32) + I_2 * mask2.astype(np.float32))  # Averaging method

    # Weighted averaging 
    epsilon = 1e-8  # Just in case, to avoid division by zero
    I_low = (I_1 * contrast_weight1 + I_2 * contrast_weight2) / (contrast_weight1 + contrast_weight2 + epsilon)

    # Gaussian Filtering
    I_low_filtered = cv2.GaussianBlur(I_low, (0, 0), sigma)
    I_high_filtered = I_high - cv2.GaussianBlur(I_high, (0, 0), sigma)

    # Fourier Filtering (NOT USED)
    #I_low_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(I_low)) * low_pass_filter(I_low.shape, 30))))
    #I_high_filtered = I_high - np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(I_high)) * low_pass_filter(I_high.shape, 30))))

    # Combine high and low-freq components and normalize
    I_hilo = I_low_filtered + alpha * I_high_filtered
    I_hilo = cv2.normalize(I_hilo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display images for visualization
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 4, 1)
    # plt.title("I_high")
    # plt.imshow(I_high, cmap='gray')

    # plt.subplot(1, 4, 2)
    # plt.title("I_high_filtered")
    # plt.imshow(I_high_filtered, cmap='gray')

    # plt.subplot(1, 4, 3)
    # plt.title("I_low_filtered")
    # plt.imshow(I_low_filtered, cmap='gray')

    # plt.subplot(1, 4, 4)
    # plt.title("I_hilo")
    # plt.imshow(I_hilo, cmap='gray')

    # plt.show()

    return I_hilo


def hilo_reconstruction2(I_high, I_structured_list, alpha=1.0, sigma=5):
    """ 
    HiLo image reconstruction with multiple structured illumination images.
    Parameters:
    - I_high: uniform illumination image.
    - I_structured_list: list of structured illumination images (ex: speckles).
    - alpha: high-freq contribution factor.
    - sigma: stdev for Gaussian filtering.
    Returns:
    - I_hilo: reconstructed HiLo image. 
    """

    I_high = I_high.astype(np.float32)
    epsilon = 1e-8

    # Initialize weighted sum and weight total
    weighted_sum = np.zeros_like(I_high, dtype=np.float32)
    weight_total = np.zeros_like(I_high, dtype=np.float32)

    for I_struct in I_structured_list:
        I_struct = I_struct.astype(np.float32)
        local_contrast = compute_local_contrast(I_struct)
        contrast_weight = cv2.normalize(local_contrast, None, 0, 1, cv2.NORM_MINMAX)

        weighted_sum += I_struct * contrast_weight
        weight_total += contrast_weight

    # Calculate I_low
    I_low = weighted_sum / (weight_total + epsilon)

    # Apply Gaussian filtering
    I_low_filtered = cv2.GaussianBlur(I_low, (0, 0), sigma)
    I_high_filtered = I_high - cv2.GaussianBlur(I_high, (0, 0), sigma)

    # Combine components for HiLo image
    I_hilo = I_low_filtered + alpha * I_high_filtered
    I_hilo = cv2.normalize(I_hilo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display images for visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title("I_high")
    plt.imshow(I_high, cmap='gray')

    plt.subplot(1, 4, 2)
    plt.title("I_high_filtered")
    plt.imshow(I_high_filtered, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title("I_low_filtered")
    plt.imshow(I_low_filtered, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title("I_hilo")
    plt.imshow(I_hilo, cmap='gray')

    plt.show()

    return I_hilo, I_high, I_high_filtered, I_low_filtered


def hilo_reconstruction(I_high, I_1, I_2, alpha=1.0, sigma=5):
    """
    HiLo image reconstruction with three input images.
    
    Parameters:
    - I_high: Image with uniform illumination (high-frequency details).
    - I_1: Image with checkerboard illumination.
    - I_2: Image with inverted checkerboard illumination.
    - alpha: Weight for high-frequency contribution.
    - sigma: Standard deviation for Gaussian filtering.

    Returns:
    - I_hilo: Reconstructed HiLo image.
    """
    # Convert images to float for calculations
    I_high = I_high.astype(np.float32)
    I_1 = I_1.astype(np.float32)
    I_2 = I_2.astype(np.float32)

    # Compute local contrast (absolute difference between checkerboard images)
    contrast = np.abs(I_1 - I_2)
    # Normalize contrast to [0,1] for weighting
    contrast_norm = cv2.normalize(contrast, None, 0, 1, cv2.NORM_MINMAX)

    # Compute the low-frequency component
    I_low = (I_1 + I_2) / 2.0  # Averaging method

    # Apply Gaussian filtering to extract low-frequency details
    I_low_filtered = contrast_norm * cv2.GaussianBlur(I_low, (0, 0), sigma)

    # Apply high-pass filtering on the high-frequency image
    I_high_filtered = I_high - cv2.GaussianBlur(I_high, (0, 0), sigma)

    # Combine low and high-frequency components
    I_hilo = I_low_filtered + alpha * I_high_filtered

    # Normalize result to [0, 255] for visualization
    I_hilo = cv2.normalize(I_hilo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    '''plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title("I_high")
    plt.imshow(I_high, cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("cI_high_filtered")
    plt.imshow(I_high_filtered, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.title("I_low_filtered")
    plt.imshow(I_low_filtered, cmap="gray")

    plt.subplot(1, 4, 4)
    plt.title("I_hilo")
    plt.imshow(I_hilo, cmap="gray")

    plt.figure()
    plt.title("I_hilo")
    plt.imshow(I_hilo, cmap="gray")

    plt.show()'''

    return I_hilo




# save_dir = "E:/surgele/tests_2603"
# path1 = os.path.join(save_dir,'E:/surgele/tests_2603/tutu_z0.tif')
# path2 = os.path.join(save_dir,'E:/surgele/tests_2603/check1_z0.tif')
# path3 = os.path.join(save_dir,'E:/surgele/tests_2603/check2_z0.tif')

# img_uniform = tif.imread("E:/surgele/tests_2603/tutu_z0.tif")
# img_check1 = tif.imread("E:/surgele/tests_2603/check1_z0.tif")
# img_check2 = tif.imread("E:/surgele/tests_2603/check2_z0.tif")
# img_uniform = np.array(Image.open('E:/surgele/tests_2603/tutu_z0.tif').convert('L'))
# img_check1 = np.array(Image.open(path2).convert('L'))
# img_check2 = np.array(Image.open(path3).convert('L'))
# print(type(img_uniform))



fichiers_tiff = sorted(glob.glob("D:\\surgele\\mai\\tests_1305\\patterns_hilo\\*.tiff"))
I_structured_list = [tif.imread(f) for f in fichiers_tiff]
img_uniform = tif.imread("D:/surgele/mai/tests_1305/perspective_centrale.tiff")

hilo_reconstruction2(img_uniform, I_structured_list, sigma=3)

# HiLo reconstruction grayscale
#I_hilo = hilo_reconstruction(img_uniform, img_check1, img_check2, alpha=1.0, sigma=5)

# HiLo reconstruction rgb
#I_hilo_rgb = hilo_reconstruction_rgb(RGBimg_uniform, RGBimg_check1, RGBimg_check2, alpha=1.5, sigma=3)

# Hilo reconstruction with contrast analysis
# I_hilo_contrast = hilo_reconstruction_contrasted(img_uniform, img_check1, img_check2, alpha=1.0, sigma=5, contrast_threshold=0.7)
