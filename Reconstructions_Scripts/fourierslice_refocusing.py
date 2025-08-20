import numpy as np
from scipy.interpolate import interpn
from scipy.fft import fftshift, ifftn, ifftshift, fftn
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial


# Provided data
depths = np.array([-84, -71, -58, -51, -38, -25, -12, 0, 26, 40, 46, 73, 79, 106])
alphas = np.array([-1.2048, -0.6920, -0.3501, -0.1450, 0.4020, 0.5388, 0.8465, 0.9832, 1.5302, 2.0089, 2.1114, 2.7610, 2.9320, 4.0943])
z_refocus = np.array([-334.06, -256.36, -204.56, -173.48, -90.6, -69.88, -23.26, -2.54, 80.34, 152.86, 168.4, 266.82, 292.72, 468.84])

# Fit a polynomial: z_refocus = f(depth)
degree = 3
coefs = Polynomial.fit(depths, z_refocus, deg=degree).convert().coef
poly_alpha_fit = Polynomial.fit(depths, alphas, deg=degree).convert()


def depth_to_z(depth):
    """Correlation between experimental depths values of refocusing and theoetical ones. Polynomial fit on the plot of (depths, z_refocused)"""
    return sum(c * depth**i for i, c in enumerate(coefs))

def depth_to_alpha(depth):
    """Return alpha given object-space depth (µm)"""
    return 0.0251 * depth + 1.0742 #poly_alpha_fit(depth)

def generate_frequency_grid(res_x, res_y):
    """
    Generate grid of frequency values for an image of size (res_x, res_y).

    Returns:
        wx, wy : 2D frequency grids
        xC, yC : Index of the zero frequency component
    """
    wx, wy = np.meshgrid(np.arange(1, res_x + 1), np.arange(1, res_y + 1))

    xC = int(np.ceil((res_x + 1) / 2)) # coordinates of the center of the frequency grid
    yC = int(np.ceil((res_y + 1) / 2))

    wx = (wx - xC) / (res_x - 1) # normalisation of the grid -> wx horizontal freq values
    wy = (yC - wy) / (res_y - 1) # same for vertical freq values

    return wx, wy, xC - 1, yC - 1  # subtract 1 for 0-based python indexing

def refocus_fourierslice(LF_FFT, res_y, res_x, alpha, is_OS):
    ny, nx = res_y, res_x # size of the image
    even_fft = 1 - np.mod([ny, nx], 2) # 1 if dimension odd, 0 if even -> to ensure symmetric cropping

    ny_pad, nx_pad, nv_pad, nu_pad = LF_FFT.shape

    # Frequency grid
    kx, ky, kx_mid, ky_mid = generate_frequency_grid(nx_pad, ny_pad) # gives normalized freq coords for spatial (x, y) domain
    ku, kv, ku_mid, kv_mid = generate_frequency_grid(nu_pad, nv_pad) # gives normalized freq coords for angular (u, v) domain
    # kx, ky, ku and kv are 2D arrays of x, y, u or v freqs
    # .._mid are for 0 freq (center)

    # Frequency scaling
    kv_alpha = (1 - alpha) * ky # sheared(cut) angular freq indices -> shift angular coords to simul refocus
    ku_alpha = (1 - alpha) * kx

    # Compute interpolation coordinates
    dKv = np.abs(kv[1, 0] - kv[0, 0]) # spacing in v
    dKu = np.abs(ku[0, 1] - ku[0, 0]) # spacing in u
    kv_alpha_idx = kv_mid - (kv_alpha / dKv) # Converts continuous frequency shift into index space
    ku_alpha_idx = (ku_alpha / dKu) + ku_mid

    # Limit indices to valid bounds
    kv_alpha_idx = np.clip(kv_alpha_idx, 0, nv_pad - 1)
    ku_alpha_idx = np.clip(ku_alpha_idx, 0, nu_pad - 1)

    # Create meshgrid for spatial dimensions
    ky_idx, kx_idx = np.meshgrid(np.arange(ny_pad), np.arange(nx_pad), indexing='ij')

    # Interpolation over 4D grid
    points = (
        np.arange(ny_pad),  # y
        np.arange(nx_pad),  # x
        np.arange(nv_pad),  # v
        np.arange(nu_pad),  # u
    )

    values = LF_FFT

    # Prepare coordinates for slicing: shape (H, W, 4)
    slice_coords = np.stack([
        ky_idx,          # y
        kx_idx,          # x
        kv_alpha_idx,    # v
        ku_alpha_idx     # u
    ], axis=-1)

    # Interpolate
    slice = interpn(points, values, slice_coords, method='linear', bounds_error=False, fill_value=0)
    slice = np.nan_to_num(slice)

    if is_OS:
        ky_idx_OS = np.linspace(0, ny_pad - 1, 2 * ny_pad)
        kx_idx_OS = np.linspace(0, nx_pad - 1, 2 * nx_pad)
        kx_OS, ky_OS = np.meshgrid(kx_idx_OS, ky_idx_OS, indexing='ij')

        # Interpolate slice to oversampled grid
        slice_OS = interpn(
            (np.arange(ny_pad), np.arange(nx_pad)),
            slice,
            np.stack([ky_OS, kx_OS], axis=-1),
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        slice_OS = np.nan_to_num(slice_OS)

        temp = fftshift(ifftn(ifftshift(slice_OS)))
        ky_mid_OS = (2 * ny_pad + 1) // 2
        kx_mid_OS = (2 * nx_pad + 1) // 2

        im_refocus = np.real(temp[
            ky_mid_OS - ny // 2 : ky_mid_OS + ny // 2 - even_fft[0],
            kx_mid_OS - nx // 2 : kx_mid_OS + nx // 2 - even_fft[1]
        ])
        im_fft = np.log(np.abs(slice_OS) + 1)
        im_fft = im_fft[
            ky_mid_OS - ny // 2 : ky_mid_OS + ny // 2 - even_fft[0],
            kx_mid_OS - nx // 2 : kx_mid_OS + nx // 2 - even_fft[1]
        ]
    else:
        temp = fftshift(ifftn(ifftshift(slice))) # 2D inverse FFT on the slice to get the refocused image
        im_refocus = np.real(temp[
            ky_mid - ny // 2 : ky_mid + ny // 2 - even_fft[0], # crop to original image size
            kx_mid - nx // 2 : kx_mid + nx // 2 - even_fft[1]
        ])
        im_fft = np.log(np.abs(slice) + 1)
        im_fft = im_fft[
            ky_mid - ny // 2 : ky_mid + ny // 2 - even_fft[0],
            kx_mid - nx // 2 : kx_mid + nx // 2 - even_fft[1]
        ]

    return im_refocus, im_fft

def refocus_test(radiance, min=100, max=100, Padding=False):
    "Performs a fourier slice refocusing and generates stacks of images and ffts"
    # Parameters
    res_y = 140
    res_x = 140
    res_v = 15
    res_u = res_v
    is_Padding = Padding
    is_OverSampling=None

    LF = radiance

    alphaS = []
    zS = []
    F = 165000 # distance between the lens(tube lens) and the sensor(MLA) = focal distance of the tube lens
    dz = 5.18 # axial resolution
    i=1
    for z in np.arange(min, max+dz, dz):
        # z_arranged = depth_to_z(z)
        # alpha = 1 + (z_arranged*(33**2))/F # dz_image = M**2 * dz_objet avec M=33 et dz_objet=(-100, ..100)
        alpha = depth_to_alpha(z)
        alphaS.append(alpha)
        zS.append(z)
        print('Slice ', i, 'z = ', z, 'alpha = ', alpha)
        i += 1

    if is_Padding:
        pad_y = round(0.1 * res_y)
        pad_x = round(0.1 * res_x)
        pad_vu = 2
        pad_width = ((pad_y, pad_y), (pad_x, pad_x), (pad_vu, pad_vu), (pad_vu, pad_vu))
        LF = np.pad(LF, pad_width, mode='constant', constant_values=0)
        
    LF_FFT = fftshift(fftn(ifftshift(LF)))

    im_refocusS =[]
    im_fftS = []
    for alpha in alphaS : # np.arange(min, max, step)
        im_refocus =[]
        im_fft = []
        im_refocus, im_fft = refocus_fourierslice(LF_FFT, res_y, res_x, alpha, is_OverSampling)
        im_refocus = 65500 * (im_refocus - np.min(im_refocus)) / (np.max(im_refocus) - np.min(im_refocus))
        im_refocusS.append(im_refocus)
        im_fftS.append(im_fft)
        
    refocus_stack = np.stack(im_refocusS)
    fft_stack = np.stack(im_fftS)

    return refocus_stack, fft_stack, zS