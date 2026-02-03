from lenstools import ShearMap, ConvergenceMap
from lenstools.utils.defaults import load_fits_default_convergence
import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

# # for every realisation in directories, make defect catalogs
# N = 128
# pixels = 512
# side_deg = 5.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# # kappa_noise_{i}_LCDM_Om02_ks_nomask_shear.fits
# for i in tqdm(range(N), desc="Processing realisations"):
#     convergence_map = ConvergenceMap.load(f"{OM02_DIR}/kappa_{i}_LCDM_Om02_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{OM02_DIR}/angles_{i}_LCDM_Om02_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{OM04_DIR}/kappa_{i}_LCDM_Om04_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{OM04_DIR}/angles_{i}_LCDM_Om04_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{S807_DIR}/kappa_{i}_LCDM_s80707210_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{S807_DIR}/angles_{i}_LCDM_s80707210_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{S809_DIR}/kappa_{i}_LCDM_s80976624_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{S809_DIR}/angles_{i}_LCDM_s80976624_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)


# # ========================================================================================
# # do for smoothed 4.67
# # for every realisation in directories, make defect catalogs
# N = 128
# pixels = 512
# side_deg = 5.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# # kappa_noise_{i}_LCDM_Om02_ks_nomask_shear.fits
# for i in tqdm(range(N), desc="Processing realisations"):
#     convergence_map = ConvergenceMap.load(f"{OM02_DIR}/kappa_s467_{i}_LCDM_Om02_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{OM02_DIR}/angles_s467_{i}_LCDM_Om02_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{OM04_DIR}/kappa_s467_{i}_LCDM_Om04_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{OM04_DIR}/angles_s467_{i}_LCDM_Om04_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{S807_DIR}/kappa_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{S807_DIR}/angles_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{S809_DIR}/kappa_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{S809_DIR}/angles_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)
    
# # do for s467 th
# N = 128
# pixels = 512
# side_deg = 5.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# # kappa_noise_{i}_LCDM_Om02_ks_nomask_shear.fits
# for i in tqdm(range(N), desc="Processing realisations"):
#     convergence_map = ConvergenceMap.load(f"{OM02_DIR}/kappa_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{OM02_DIR}/angles_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{OM04_DIR}/kappa_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{OM04_DIR}/angles_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{S807_DIR}/kappa_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{S807_DIR}/angles_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)

#     convergence_map = ConvergenceMap.load(f"{S809_DIR}/kappa_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz", format="npz")
#     shear_map = ShearMap.fromConvergence(convergence_map)
#     shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#     np.savez(f"{S809_DIR}/angles_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz", 
#              shear_angles=shear_angles, 
#              kappa=convergence_map.data,
#              gamm1=shear_map.data[0],
#              gamm2=shear_map.data[1],
#              angle_deg=angle_deg)


# N = 1101
# pixels = 1024
# side_deg = 10.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         convergence_map = ConvergenceMap.load(f"{SLICS_DIR}/kappa_{i}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
#         shear_map = ShearMap.fromConvergence(convergence_map)
#         shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#         np.savez(f"{SLICS_DIR}/angles_{i}_LCDM_SLICS_ks_nomask_shear.npz", 
#                  shear_angles=shear_angles, 
#                  kappa=convergence_map.data,
#                  gamm1=shear_map.data[0],
#                  gamm2=shear_map.data[1],
#                  angle_deg=angle_deg)
#         for subfield in range(4):
#             convergence_subfield = ConvergenceMap.load(f"{SLICS_DIR}/kappa_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
#             shear_map_subfield = ShearMap.fromConvergence(convergence_subfield) 
#             shear_angles_subfield = 0.5 * np.arctan2(shear_map_subfield.data[1], shear_map_subfield.data[0])
#             x_start = (subfield % 2) * 512
#             y_start = (subfield // 2) * 512
#             angle_deg_subfield = angle_deg[:, y_start:y_start+512, x_start:x_start+512]
#             np.savez(f"{SLICS_DIR}/angles_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", 
#                      shear_angles=shear_angles_subfield, 
#                      kappa=convergence_subfield.data,
#                      gamm1=shear_map_subfield.data[0],
#                      gamm2=shear_map_subfield.data[1],
#                      angle_deg=angle_deg_subfield)
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")

# N = 1101
# pixels = 1024
# side_deg = 10.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         convergence_map = ConvergenceMap.load(f"{SLICS_DIR}/kappa_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
#         shear_map = ShearMap.fromConvergence(convergence_map)
#         shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#         np.savez(f"{SLICS_DIR}/angles_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz", 
#                  shear_angles=shear_angles, 
#                  kappa=convergence_map.data,
#                  gamm1=shear_map.data[0],
#                  gamm2=shear_map.data[1],
#                  angle_deg=angle_deg)
#         # split into 4 sub-fields of 512x512 pixels each
#         for subfield in range(4):
#             x_start = (subfield % 2) * 512
#             y_start = (subfield // 2) * 512
#             convergence_subfield = ConvergenceMap.load(f"{SLICS_DIR}/kappa_s467_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
#             shear_map_subfield = ShearMap.fromConvergence(convergence_subfield) 
#             shear_angles_subfield = 0.5 * np.arctan2(shear_map_subfield.data[1], shear_map_subfield.data[0])
#             angle_deg_subfield = angle_deg[:, y_start:y_start+512, x_start:x_start+512]
#             np.savez(f"{SLICS_DIR}/angles_s467_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", 
#                      shear_angles=shear_angles_subfield, 
#                      kappa=convergence_subfield.data,
#                      gamm1=shear_map_subfield.data[0],
#                      gamm2=shear_map_subfield.data[1],
#                      angle_deg=angle_deg_subfield)
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")

# N = 1101
# pixels = 1024
# side_deg = 10.0
# angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
# angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         convergence_map = ConvergenceMap.load(f"{SLICS_DIR}/kappa_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
#         shear_map = ShearMap.fromConvergence(convergence_map)
#         shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
#         np.savez(f"{SLICS_DIR}/angles_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz", 
#                  shear_angles=shear_angles, 
#                  kappa=convergence_map.data,
#                  gamm1=shear_map.data[0],
#                  gamm2=shear_map.data[1],
#                  angle_deg=angle_deg)
#         # split into 4 sub-fields of 512x512 pixels each
#         for subfield in range(4):
#             x_start = (subfield % 2) * 512
#             y_start = (subfield // 2) * 512
#             convergence_subfield = ConvergenceMap.load(f"{SLICS_DIR}/kappa_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
#             shear_map_subfield = ShearMap.fromConvergence(convergence_subfield) 
#             shear_angles_subfield = 0.5 * np.arctan2(shear_map_subfield.data[1], shear_map_subfield.data[0])
#             angle_deg_subfield = angle_deg[:, y_start:y_start+512, x_start:x_start+512]
#             np.savez(f"{SLICS_DIR}/angles_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", 
#                      shear_angles=shear_angles_subfield, 
#                      kappa=convergence_subfield.data,
#                      gamm1=shear_map_subfield.data[0],
#                      gamm2=shear_map_subfield.data[1],
#                      angle_deg=angle_deg_subfield)
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")

# ======================
# define Mexican-hat kernel
# ======================
def mexican_hat_kernel(theta_pix, size=None):
    """
    theta_pix: sigma in pixels
    size: kernel size (odd integer)
    """
    radius = size // 2
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    r2 = x**2 + y**2
    U = (1 - r2 / (2*theta_pix**2)) * np.exp(-r2 / (2*theta_pix**2))
    U /= (2 * np.pi * theta_pix**2)  
    return U

# ======================
# shear smoothing function
# ======================
def smooth_shear(shear_map, kernel):
    """
    Smooth the shear map using the provided kernel.
    shear_map: ShearMap object
    kernel: 2D numpy array
    """
    smoothed = np.zeros_like(shear_map.data)
    for i in range(2):  # gamma1 and gamma2
        smoothed[i] = convolve2d(shear_map.data[i], kernel, mode='same', boundary='symm')
    shear_map.data[:] = smoothed
    return shear_map

sigma_pix = 8.0
kernel = mexican_hat_kernel(sigma_pix, size=25)

# ========================================================================================
# Shear smoothed by 8 pixels
# ========================================================================================
N = 128
pixels = 512
side_deg = 5.0
angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
sigma_pix = 4.0

for i in tqdm(range(N), desc="Processing Shear Smooth 8pix (Params)"):
    # Om02
    try:
        convergence_map = ConvergenceMap.load(f"{OM02_DIR}/kappa_{i}_LCDM_Om02_ks_nomask_shear.npz", format="npz")
        shear_map = ShearMap.fromConvergence(convergence_map)
        shear_map = smooth_shear(shear_map, kernel)
        convergence_map_smooth = shear_map.convergence()
        shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
        np.savez(f"{OM02_DIR}/angles_s234_fromshear_{i}_LCDM_Om02_ks_nomask_shear.npz", 
                 shear_angles=shear_angles, 
                 kappa=convergence_map_smooth.data,
                 gamm1=shear_map.data[0],
                 gamm2=shear_map.data[1],
                 angle_deg=angle_deg)
    except FileNotFoundError:
        pass

    # Om04
    try:
        convergence_map = ConvergenceMap.load(f"{OM04_DIR}/kappa_{i}_LCDM_Om04_ks_nomask_shear.npz", format="npz")
        shear_map = ShearMap.fromConvergence(convergence_map)
        shear_map = smooth_shear(shear_map, kernel)
        convergence_map_smooth = shear_map.convergence()
        shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
        np.savez(f"{OM04_DIR}/angles_s234_fromshear_{i}_LCDM_Om04_ks_nomask_shear.npz", 
                 shear_angles=shear_angles, 
                 kappa=convergence_map_smooth.data,
                 gamm1=shear_map.data[0],
                 gamm2=shear_map.data[1],
                 angle_deg=angle_deg)
    except FileNotFoundError:
        pass

    # S807
    try:
        convergence_map = ConvergenceMap.load(f"{S807_DIR}/kappa_{i}_LCDM_s80707210_ks_nomask_shear.npz", format="npz")
        shear_map = ShearMap.fromConvergence(convergence_map)
        shear_map = smooth_shear(shear_map, kernel)
        convergence_map_smooth = shear_map.convergence()
        shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
        np.savez(f"{S807_DIR}/angles_s234_fromshear_{i}_LCDM_s80707210_ks_nomask_shear.npz", 
                 shear_angles=shear_angles, 
                 kappa=convergence_map_smooth.data,
                 gamm1=shear_map.data[0],
                 gamm2=shear_map.data[1],
                 angle_deg=angle_deg)
    except FileNotFoundError:
        pass

    # S809
    try:
        convergence_map = ConvergenceMap.load(f"{S809_DIR}/kappa_{i}_LCDM_s80976624_ks_nomask_shear.npz", format="npz")
        shear_map = ShearMap.fromConvergence(convergence_map)
        shear_map = smooth_shear(shear_map, kernel)
        convergence_map_smooth = shear_map.convergence()
        shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
        np.savez(f"{S809_DIR}/angles_s234_fromshear_{i}_LCDM_s80976624_ks_nomask_shear.npz", 
                 shear_angles=shear_angles, 
                 kappa=convergence_map_smooth.data,
                 gamm1=shear_map.data[0],
                 gamm2=shear_map.data[1],
                 angle_deg=angle_deg)
    except FileNotFoundError:
        pass


N = 1101
pixels = 1024
side_deg = 10.0
angle_grid = np.linspace(0, side_deg, pixels) + 0.5 * (side_deg / pixels)
angle_deg = np.array(np.meshgrid(angle_grid, angle_grid))
for i in tqdm(range(N), desc="Processing SLICS Shear Smooth 8pix"):
    try:
        convergence_map = ConvergenceMap.load(f"{SLICS_DIR}/kappa_{i}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
        shear_map = ShearMap.fromConvergence(convergence_map)
        shear_map = smooth_shear(shear_map, kernel)
        convergence_map_smooth = shear_map.convergence()
        shear_angles = 0.5 * np.arctan2(shear_map.data[1], shear_map.data[0])
        np.savez(f"{SLICS_DIR}/angles_s234_fromshear_{i}_LCDM_SLICS_ks_nomask_shear.npz", 
                 shear_angles=shear_angles, 
                 kappa=convergence_map_smooth.data,
                 gamm1=shear_map.data[0],
                 gamm2=shear_map.data[1],
                 angle_deg=angle_deg)
        for subfield in range(4):
            convergence_subfield = ConvergenceMap.load(f"{SLICS_DIR}/kappa_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", format="npz")
            shear_map_subfield = ShearMap.fromConvergence(convergence_subfield) 
            shear_map_subfield = smooth_shear(shear_map_subfield, kernel)
            convergence_subfield_smooth = shear_map_subfield.convergence()
            shear_angles_subfield = 0.5 * np.arctan2(shear_map_subfield.data[1], shear_map_subfield.data[0])
            x_start = (subfield % 2) * 512
            y_start = (subfield // 2) * 512
            angle_deg_subfield = angle_deg[:, y_start:y_start+512, x_start:x_start+512]
            np.savez(f"{SLICS_DIR}/angles_s234_fromshear_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz", 
                     shear_angles=shear_angles_subfield, 
                     kappa=convergence_subfield_smooth.data,
                     gamm1=shear_map_subfield.data[0],
                     gamm2=shear_map_subfield.data[1],
                     angle_deg=angle_deg_subfield)
    except FileNotFoundError:
        print(f"File for realisation {i} not found, skipping.")