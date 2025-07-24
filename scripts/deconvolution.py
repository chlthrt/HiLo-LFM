import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.signal
from skimage import io
from skimage.restoration import richardson_lucy
import tifffile as tif


img_path = "D:/surgele/new_billes/zstack3.tif"
psf_path = "D:/surgele/new_billes/zstack3_crop.tif"

img = io.imread(img_path)  # Shape: (z, y, x)
psf = io.imread(psf_path)      # Should be similar shape


# psf = psf / psf.sum()


############ Richardson Lucy ##############

deconvolved = richardson_lucy(img, psf, num_iter=10)

io.imsave("D:/surgele/new_billes/deconvolved_zstack_py1.tif", deconvolved.astype(np.float32))

################### Wiener ####################

def estimate_snr(img):
    mean_signal = np.mean(img)
    std_noise = np.std(img)
    return mean_signal / std_noise

def wiener_deconvolution(img, psf, snr):
    H = np.fft.fft2(psf, s=img.shape)  # on zero-pad la PSF pour fit les sizes
    G = np.fft.fft2(img)
    denominator = H * np.conj(H) + (1 / snr)**2
    F = G * np.conj(H) / denominator
    deconvolved = np.real(np.fft.ifft2(F))
    return deconvolved

# wiener = wiener_deconvolution(img_norm, psf_norm, estimate_snr(img_norm))

###################### Plot ######################

print(deconvolved.shape)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Initial")
plt.imshow(img[40], cmap='gray')
plt.subplot(1, 3, 2)
plt.title("RL")
plt.imshow(deconvolved[40], cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Wiener")
# plt.imshow(wiener, cmap='gray')

plt.show()



