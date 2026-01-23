# transfer convergence fits to npz format
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, uniform_filter

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

# # 512x512 pixels, 5x5 deg^2, create RA (0, 5) DEC (0, 5) angle grid

# # first, have a look at the data
# with fits.open(f"{OM02_DIR}/kappa_noise_0_LCDM_Om02_ks_nomask_shear.fits") as hdul:
#     convergence_data = hdul[0].data[0] - hdul[0].data[1]
#     convergence_smoothed = gaussian_filter(hdul[0].data[0], sigma=8)  # apply Gaussian smoothing
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.imshow(convergence_data, origin='lower', cmap='viridis')
# plt.colorbar()
# plt.title("No Noise")
# plt.subplot(1,2,2)
# plt.imshow(convergence_smoothed, origin='lower', cmap='viridis')
# plt.colorbar()
# plt.title("With Noise, smoothed")
# plt.savefig("outfiles/convergence_example.png", dpi=300)


# N = 128
# pixels = 512
# side_deg = 5.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# for i in tqdm(range(N), desc="Processing realisations"):
#     with fits.open(f"{OM02_DIR}/kappa_noise_{i}_LCDM_Om02_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = uniform_filter(convergence_data, size=8)
#     np.savez(f"{OM02_DIR}/kappa_{i}_LCDM_Om02_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{OM02_DIR}/kappa_s467_{i}_LCDM_Om02_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{OM02_DIR}/kappa_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)
    
#     with fits.open(f"{OM04_DIR}/kappa_noise_{i}_LCDM_Om04_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = uniform_filter(convergence_data, size=8)
#     np.savez(f"{OM04_DIR}/kappa_{i}_LCDM_Om04_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{OM04_DIR}/kappa_s467_{i}_LCDM_Om04_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{OM04_DIR}/kappa_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)
    
#     with fits.open(f"{S807_DIR}/kappa_noise_{i}_LCDM_s80707210_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = uniform_filter(convergence_data, size=8)
#     np.savez(f"{S807_DIR}/kappa_{i}_LCDM_s80707210_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{S807_DIR}/kappa_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{S807_DIR}/kappa_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)

#     with fits.open(f"{S809_DIR}/kappa_noise_{i}_LCDM_s80976624_ks_nomask_shear.fits") as hdul:
#         convergence_data = hdul[0].data[0]  # add noise
#         convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
#         convergence_smoothed_th = uniform_filter(convergence_data, size=8)
#     np.savez(f"{S809_DIR}/kappa_{i}_LCDM_s80976624_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
#     np.savez(f"{S809_DIR}/kappa_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
#     np.savez(f"{S809_DIR}/kappa_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)

N = 1101
pixels = 1024
side_deg = 10.0
angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
count = 0
for i in tqdm(range(1000, N), desc="Processing realisations"):
    try:
        with fits.open(f"{SLICS_DIR}/kappa_noise_GalCatalog_LOS_cone{i}.fits_s333{i}_zmin0.0_zmax3.0_sys_3.fits_ks_nomask_shear.fits") as hdul:
            convergence_data = hdul[0].data[0]  # add noise
            convergence_smoothed = gaussian_filter(convergence_data, sigma=8)  # apply Gaussian smoothing
            convergence_smoothed_th = uniform_filter(convergence_data, size=8)
        np.savez(f"{SLICS_DIR}/kappa_{i}_LCDM_SLICS_ks_nomask_shear.npz", convergence_data, angle_deg=angle_deg)
        np.savez(f"{SLICS_DIR}/kappa_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed, angle_deg=angle_deg)
        np.savez(f"{SLICS_DIR}/kappa_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed_th, angle_deg=angle_deg)
        # split into 4 sub-fields of 512x512 pixels each
        for subfield in range(4):
            x_start = (subfield % 2) * 512
            y_start = (subfield // 2) * 512
            convergence_subfield = convergence_data[y_start:y_start+512, x_start:x_start+512]
            convergence_smoothed_subfield = convergence_smoothed[y_start:y_start+512, x_start:x_start+512]
            convergence_smoothed_th_subfield = convergence_smoothed_th[y_start:y_start+512, x_start:x_start+512]
            angle_deg_subfield = angle_deg[:, y_start:y_start+512, x_start:x_start+512]
            np.savez(f"{SLICS_DIR}/kappa_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", convergence_subfield, angle_deg=angle_deg_subfield)
            np.savez(f"{SLICS_DIR}/kappa_s467_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed_subfield, angle_deg=angle_deg_subfield)
            np.savez(f"{SLICS_DIR}/kappa_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", convergence_smoothed_th_subfield, angle_deg=angle_deg_subfield)
        count += 1
    except FileNotFoundError:
        print(f"File for realisation {i} not found, skipping.")

print(f"Processed {count} realisations.")