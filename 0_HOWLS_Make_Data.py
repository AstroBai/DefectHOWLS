# transfer convergence fits to npz format
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, uniform_filter, convolve

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

# Create Circular Top-Hat Kernel
radius = 8
y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 <= radius**2
tophat_kernel = mask.astype(float)
tophat_kernel /= tophat_kernel.sum()

# 512x512 pixels, 5x5 deg^2, create RA (0, 5) DEC (0, 5) angle grid


def measure_power_spectrum(map_data, pixel_size_deg):
    """
    Measure the angular power spectrum of a 2D convergence map.
    """
    N = map_data.shape[0]
    pixel_size_rad = np.radians(pixel_size_deg)
    
    # FFT
    map_fft = np.fft.fft2(map_data)
    map_fft = np.fft.fftshift(map_fft)  # Shift zero freq to center
    
    # k-space grid (ell space)
    freq = np.fft.fftfreq(N, d=pixel_size_rad)
    freq = np.fft.fftshift(freq)
    lx, ly = np.meshgrid(freq * 2 * np.pi, freq * 2 * np.pi)
    l = np.sqrt(lx**2 + ly**2)
    
    # Power spectrum normalization
    # P(l) = |fft|^2 * (Area / N^4)
    # Area in radians^2
    total_area_rad = (N * pixel_size_rad)**2
    P2D = np.abs(map_fft)**2 * (total_area_rad / N**4)

    # Azimuthal averaging
    l_min = np.min(l[l>0])
    l_max = np.max(l)
    n_bins = 100 # Number of bins
    l_bins = np.logspace(np.log10(l_min), np.log10(l_max), n_bins+1)
    
    ell_vals = []
    Cl_vals = []
    
    for i in range(n_bins):
        mask = (l >= l_bins[i]) & (l < l_bins[i+1])
        if np.sum(mask) > 0:
            ell_vals.append(np.mean(l[mask]))
            Cl_vals.append(np.mean(P2D[mask]))
            
    return np.array(ell_vals), np.array(Cl_vals)

# first, have a look at the data
with fits.open(f"{OM02_DIR}/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits") as hdul:
    convergence_data = hdul[0].data[0] # no noise
    noise = hdul[0].data[1]  # apply Gaussian smoothing

# Measure Power Spectrum
pixel_size = 5.0 / 512 # deg
ell, Cl = measure_power_spectrum(convergence_data, pixel_size)
ell_n, Cl_n = measure_power_spectrum(noise, pixel_size)
ell_smooth, Cl_smooth = measure_power_spectrum(gaussian_filter(convergence_data, sigma=8), pixel_size)
ell_n_smooth, Cl_n_smooth = measure_power_spectrum(gaussian_filter(noise, sigma=8), pixel_size)

print(np.mean(convergence_data))
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 5))

plt.subplot(1, 4, 1)
plt.imshow(convergence_data, origin='lower', cmap='viridis')
plt.colorbar()
plt.title("With Noise")

plt.subplot(1, 4, 2)
plt.imshow(noise, origin='lower', cmap='viridis')
plt.colorbar()
plt.title("Noise")

plt.subplot(1, 4, 3)
plt.loglog(ell, Cl, label='With Noise')
plt.loglog(ell_n, Cl_n, label='Noise')
plt.loglog(ell, Cl-Cl_n, label='Difference', linestyle='--')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell$')
plt.title("Angular Power Spectrum")
plt.legend()

plt.subplot(1, 4, 4)
plt.loglog(ell_smooth, Cl_smooth, label='Smoothed With Noise')
plt.loglog(ell_n_smooth, Cl_n_smooth, label='Smoothed Noise')
plt.loglog(ell_smooth, Cl_smooth-Cl_n_smooth, label='Smoothed Difference', linestyle='--')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell$')
plt.title("Smoothed Angular Power Spectrum")
plt.legend()

plt.savefig("outfiles/convergence_example.png", dpi=300)
print("noise cl mean:", np.mean(Cl_n))

# N = 128
# pixels = 512
# side_deg = 5.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# for i in tqdm(range(N), desc="Processing realisations"):
#     with fits.open(f"{OM02_DIR}/kappa_noise_{i}_LCDM_Om02_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = convolve(convergence_data, tophat_kernel)
#     np.savez(f"{OM02_DIR}/kappa_{i}_LCDM_Om02_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{OM02_DIR}/kappa_s467_{i}_LCDM_Om02_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{OM02_DIR}/kappa_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)
    
#     with fits.open(f"{OM04_DIR}/kappa_noise_{i}_LCDM_Om04_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = convolve(convergence_data, tophat_kernel)
#     np.savez(f"{OM04_DIR}/kappa_{i}_LCDM_Om04_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{OM04_DIR}/kappa_s467_{i}_LCDM_Om04_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{OM04_DIR}/kappa_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)
    
#     with fits.open(f"{S807_DIR}/kappa_noise_{i}_LCDM_s80707210_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = convolve(convergence_data, tophat_kernel)
#     np.savez(f"{S807_DIR}/kappa_{i}_LCDM_s80707210_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{S807_DIR}/kappa_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{S807_DIR}/kappa_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)

#     with fits.open(f"{S809_DIR}/kappa_noise_{i}_LCDM_s80976624_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = convolve(convergence_data, tophat_kernel)
#     np.savez(f"{S809_DIR}/kappa_{i}_LCDM_s80976624_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{S809_DIR}/kappa_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{S809_DIR}/kappa_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)

# N = 1101
# pixels = 1024
# side_deg = 10.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# count = 0
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         with fits.open(f"{SLICS_DIR}/kappa_noise_GalCatalog_LOS_cone{i}.fits_s333{i}_zmin0.0_zmax3.0_sys_3.fits_ks_nomask_shear.fits") as hdul:
#             convergence_data = hdul[0].data[0]  # add noise
#             convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#             convergence_smoothed_th = convolve(convergence_data, tophat_kernel)
#         np.savez(f"{SLICS_DIR}/kappa_{i}_LCDM_SLICS_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#         np.savez(f"{SLICS_DIR}/kappa_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#         np.savez(f"{SLICS_DIR}/kappa_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)
#         # split into 4 sub-fields of 512x512 pixels each
#         for subfield in range(4):
#             x_start = (subfield % 2) * 512
#             y_start = (subfield // 2) * 512
#             convergence_subfield = convergence_data[y_start:y_start+512, x_start:x_start+512]
#             convergence_smoothed_subfield = convergence_smoothed[y_start:y_start+512, x_start:x_start+512]
#             convergence_smoothed_th_subfield = convergence_smoothed_th[y_start:y_start+512, x_start:x_start+512]
#             angle_deg_subfield = angle_deg[:, y_start:y_start+512, x_start:x_start+512]
#             np.savez(f"{SLICS_DIR}/kappa_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", convergence_subfield, angle_deg=angle_deg_subfield)
#             np.savez(f"{SLICS_DIR}/kappa_s467_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed_subfield, angle_deg=angle_deg_subfield)
#             np.savez(f"{SLICS_DIR}/kappa_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed_th_subfield, angle_deg=angle_deg_subfield)
#         count += 1
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")

# print(f"Processed {count} realisations.")