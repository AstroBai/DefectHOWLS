from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from DefectFinder import DefectFinder
from mpl_toolkits.axes_grid1 import make_axes_locatable 

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

# --- Configuration for Zoom ---
N_pix_total = 1024
FoV_total = 10.0 # degrees
FoV_zoom = 2   # degrees
seed = 997

# Calculate pixel limit for the zoom
limit_pix = int((FoV_zoom / FoV_total) * N_pix_total)
print(f"Zooming into first {limit_pix}x{limit_pix} pixels ({FoV_zoom}x{FoV_zoom} deg)")

# 1. Load Convergence Data (Background Image)
print("Loading convergence data...")
with fits.open(f"{SLICS_DIR}/kappa_noise_GalCatalog_LOS_cone{seed}.fits_s333{seed}_zmin0.0_zmax3.0_sys_3.fits_ks_nomask_shear.fits") as hdul:
    convergence_data = hdul[0].data[0]  # add noise
    # Apply Gaussian smoothing (sigma=8 pixels)
    convergence_smoothed = gaussian_filter(convergence_data, sigma=8)

# 2. Load DefectFinder Data (Shear & Defects)
print("Loading defect data...")
df = DefectFinder(f"{SLICS_DIR}/angles_s467_{seed}_LCDM_SLICS_ks_nomask_shear.npz")
df.find_defects(savepath=f"{SLICS_DIR}/defects_s467_{seed}_LCDM_SLICS_ks_nomask_shear.npz")
print(df.number())

# Extract fields
phi = df.phi
shear_amp = df.shear_amp
defects = df.defects

# Calculate shear orientation vectors
ux = np.cos(phi)
uy = np.sin(phi)

# --- Slicing Data for Zoom ---
# Slice images
conv_zoom = convergence_smoothed[0:limit_pix, 0:limit_pix]
shear_zoom = shear_amp[0:limit_pix, 0:limit_pix]
phi_zoom = phi[0:limit_pix, 0:limit_pix] # Slice phase field
ux_zoom = ux[0:limit_pix, 0:limit_pix]
uy_zoom = uy[0:limit_pix, 0:limit_pix]

# Filter defects within the range
defects_zoom = {}
for pix, wnum in defects.items():
    if isinstance(pix, (int, np.integer)):
        r, c = np.unravel_index(pix, (N_pix_total, N_pix_total))
    else:
        r, c = pix
    
    # Check if defect is inside the zoom window
    if 0 <= r < limit_pix and 0 <= c < limit_pix:
        defects_zoom[(r, c)] = wnum

# 3. Plotting
print("Plotting...")
# Changed to 3 subplots, increased figure width
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5)) 
extent = [0, FoV_zoom, 0, FoV_zoom] # Set axis to degrees

# --- Subplot 1: Convergence with Shear Orientation ---
im1 = ax1.imshow(conv_zoom, origin='lower', cmap='RdYlBu_r', extent=extent)

# 使用 make_axes_locatable 创建与 ax1 高度一致的 colorbar
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im1, cax=cax1, label=r'Smoothed Convergence $\kappa$')

# Quiver (Downsampled)
step = 8
x_deg = np.linspace(0, FoV_zoom, limit_pix)
y_deg = np.linspace(0, FoV_zoom, limit_pix)
X, Y = np.meshgrid(x_deg, y_deg)

ax1.quiver(X[::step, ::step], Y[::step, ::step], 
          ux_zoom[::step, ::step], uy_zoom[::step, ::step], 
          angles='xy', scale_units='xy',
          color='white', width=0.004, alpha=0.6,
          headwidth=0, headlength=0, headaxislength=0, pivot='mid')

# Scatter Defects
for (r, c), wnum in defects_zoom.items():
    x_pos = (c / limit_pix) * FoV_zoom
    y_pos = (r / limit_pix) * FoV_zoom
    
    if wnum > 0:
        color = 'red'
        sign = '+'
        label = 'Positive'
    else:
        color = 'royalblue'
        sign = '-'
        label = 'Negative'
        
    current_labels = [l.get_label() for l in ax1.collections]
    lbl = label if label not in current_labels else ""
    
    ax1.scatter(x_pos, y_pos, c=color, marker='o', s=100, edgecolors='white', linewidth=1, label=lbl, zorder=10)
    ax1.text(x_pos, y_pos, sign, ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=11)

ax1.set_title(f'Convergence ({FoV_zoom}x{FoV_zoom}'+r' ${\rm deg}^2$)')
ax1.set_xlabel('RA [deg]')
ax1.set_ylabel('Dec [deg]')

# --- Subplot 2: Shear Amplitude with Orientation ---
im2 = ax2.imshow(shear_zoom, origin='lower', cmap='plasma', extent=extent)

# 使用 make_axes_locatable 创建与 ax2 高度一致的 colorbar
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im2, cax=cax2, label=r'Shear Amplitude $|\gamma|$')

ax2.quiver(X[::step, ::step], Y[::step, ::step], 
          ux_zoom[::step, ::step], uy_zoom[::step, ::step], 
          angles='xy', scale_units='xy',
          color='white', width=0.004, alpha=0.6,
          headwidth=0, headlength=0, headaxislength=0, pivot='mid')

for (r, c), wnum in defects_zoom.items():
    x_pos = (c / limit_pix) * FoV_zoom
    y_pos = (r / limit_pix) * FoV_zoom
    
    if wnum > 0:
        color = 'red'
        sign = '+'
    else:
        color = 'royalblue'
        sign = '-'
        
    ax2.scatter(x_pos, y_pos, c=color, marker='o', s=100, edgecolors='white', linewidth=1, zorder=10)
    ax2.text(x_pos, y_pos, sign, ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=11)

ax2.set_title(f'Shear Amplitude ({FoV_zoom}x{FoV_zoom}'+r' ${\rm deg}^2$)')
ax2.set_xlabel('RA [deg]')
ax2.set_ylabel('Dec [deg]')

# --- Subplot 3: Shear Phase (No Quiver) ---
# Using 'twilight' colormap which is good for cyclic phase data
im3 = ax3.imshow(phi_zoom, origin='lower', cmap='twilight', extent=extent)

# 使用 make_axes_locatable 创建与 ax3 高度一致的 colorbar
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.1)
cbar3 = plt.colorbar(im3, cax=cax3, label=r'Shear Phase $\varphi$')

# Define ticks for pi multiples
# Assuming phi is in radians. Adjust range if your phi is defined differently (e.g. 0 to pi)
# Common range for shear angle is [-pi/2, pi/2] or [-pi, pi]
# Let's assume [-pi, pi] for general phase, or check data range.
# If it's shear angle, it's usually modulo pi.

# Set ticks at -pi, -pi/2, 0, pi/2, pi
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']

# If the data range is smaller (e.g. shear is often defined in [-pi/2, pi/2])
if phi_zoom.min() > -np.pi/2 - 0.1 and phi_zoom.max() < np.pi/2 + 0.1:
    ticks = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    tick_labels = [r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$']

cbar3.set_ticks(ticks)
cbar3.set_ticklabels(tick_labels)

for (r, c), wnum in defects_zoom.items():
    x_pos = (c / limit_pix) * FoV_zoom
    y_pos = (r / limit_pix) * FoV_zoom
    
    if wnum > 0:
        color = 'red'
        sign = '+'
    else:
        color = 'royalblue'
        sign = '-'
        
    ax3.scatter(x_pos, y_pos, c=color, marker='o', s=100, edgecolors='white', linewidth=1, zorder=10)
    ax3.text(x_pos, y_pos, sign, ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=11)

ax3.set_title(f'Shear Phase ({FoV_zoom}x{FoV_zoom}'+r' ${\rm deg}^2$)')
ax3.set_xlabel('RA [deg]')
ax3.set_ylabel('Dec [deg]')

plt.tight_layout(pad=0.3)
plt.savefig("outfiles/Demo_Defects.png", dpi=300, bbox_inches='tight')
print("Saved plot to outfiles/Demo_Defects.png")

# --- Version without defect markers (no circles, no +/− signs) ---
print("Plotting version without defect markers...")
fig2, (ax1_nd, ax2_nd, ax3_nd) = plt.subplots(1, 3, figsize=(18, 5))
extent = [0, FoV_zoom, 0, FoV_zoom]

# Subplot 1: Convergence with Shear Orientation (no defect markers)
im1_nd = ax1_nd.imshow(conv_zoom, origin='lower', cmap='RdYlBu_r', extent=extent)
divider1_nd = make_axes_locatable(ax1_nd)
cax1_nd = divider1_nd.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im1_nd, cax=cax1_nd, label=r'Smoothed Convergence $\kappa$')

step = 8
x_deg = np.linspace(0, FoV_zoom, limit_pix)
y_deg = np.linspace(0, FoV_zoom, limit_pix)
X, Y = np.meshgrid(x_deg, y_deg)

ax1_nd.quiver(X[::step, ::step], Y[::step, ::step], 
              ux_zoom[::step, ::step], uy_zoom[::step, ::step], 
              angles='xy', scale_units='xy',
              color='white', width=0.004, alpha=0.6,
              headwidth=0, headlength=0, headaxislength=0, pivot='mid')

ax1_nd.set_title(f'Convergence ({FoV_zoom}x{FoV_zoom}'+r' ${\rm deg}^2$)')
ax1_nd.set_xlabel('RA [deg]')
ax1_nd.set_ylabel('Dec [deg]')

# Subplot 2: Shear Amplitude with Orientation (no defect markers)
im2_nd = ax2_nd.imshow(shear_zoom, origin='lower', cmap='plasma', extent=extent)
divider2_nd = make_axes_locatable(ax2_nd)
cax2_nd = divider2_nd.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im2_nd, cax=cax2_nd, label=r'Shear Amplitude $|\gamma|$')

ax2_nd.quiver(X[::step, ::step], Y[::step, ::step], 
              ux_zoom[::step, ::step], uy_zoom[::step, ::step], 
              angles='xy', scale_units='xy',
              color='white', width=0.004, alpha=0.6,
              headwidth=0, headlength=0, headaxislength=0, pivot='mid')

ax2_nd.set_title(f'Shear Amplitude ({FoV_zoom}x{FoV_zoom}'+r' ${\rm deg}^2$)')
ax2_nd.set_xlabel('RA [deg]')
ax2_nd.set_ylabel('Dec [deg]')

# Subplot 3: Shear Phase (no defect markers)
im3_nd = ax3_nd.imshow(phi_zoom, origin='lower', cmap='twilight', extent=extent)
divider3_nd = make_axes_locatable(ax3_nd)
cax3_nd = divider3_nd.append_axes("right", size="5%", pad=0.1)
cbar3_nd = plt.colorbar(im3_nd, cax=cax3_nd, label=r'Shear Phase $\varphi$')

ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
if phi_zoom.min() > -np.pi/2 - 0.1 and phi_zoom.max() < np.pi/2 + 0.1:
    ticks = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    tick_labels = [r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$']

cbar3_nd.set_ticks(ticks)
cbar3_nd.set_ticklabels(tick_labels)

ax3_nd.set_title(f'Shear Phase ({FoV_zoom}x{FoV_zoom}'+r' ${\rm deg}^2$)')
ax3_nd.set_xlabel('RA [deg]')
ax3_nd.set_ylabel('Dec [deg]')

plt.tight_layout(pad=0.3)
plt.savefig("outfiles/Demo_Defects_nomarkers.png", dpi=300, bbox_inches='tight')
print("Saved plot to outfiles/Demo_Defects_nomarkers.png")

# --- Minimal single-field panels (no labels/axes/colorbars/defects) ---
print("Plotting minimalist single-field panels...")
minimal_panels = [
    (conv_zoom, 'RdYlBu_r', "outfiles/Demo_Defects_conv_clean.png", True),
    (shear_zoom, 'plasma', "outfiles/Demo_Defects_shear_clean.png", True),
    (phi_zoom, 'twilight', "outfiles/Demo_Defects_phase_clean.png", False),
]

# Recreate coarse sampling grid so quiver white lines remain on relevant panels
step_clean = 8
x_deg_clean = np.linspace(0, FoV_zoom, limit_pix)
y_deg_clean = np.linspace(0, FoV_zoom, limit_pix)
X_clean, Y_clean = np.meshgrid(x_deg_clean, y_deg_clean)

for data, cmap, out_path, add_quiver in minimal_panels:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data, origin='lower', cmap=cmap, extent=extent)
    if add_quiver:
        ax.quiver(X_clean[::step_clean, ::step_clean],
                  Y_clean[::step_clean, ::step_clean],
                  ux_zoom[::step_clean, ::step_clean],
                  uy_zoom[::step_clean, ::step_clean],
                  angles='xy', scale_units='xy',
                  color='white', width=0.004, alpha=0.7,
                  headwidth=0, headlength=0, headaxislength=0, pivot='mid')
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

# --- Single-field panels with colorbars ---
print("Plotting single-field panels with colorbars...")
colorbar_panels = [
    (conv_zoom, 'RdYlBu_r', "outfiles/Demo_Defects_conv_cb.png", True, r'$\kappa$'),
    (shear_zoom, 'plasma', "outfiles/Demo_Defects_shear_cb.png", True, r'$|\gamma|$'),
    (phi_zoom, 'twilight', "outfiles/Demo_Defects_phase_cb.png", False, r'$\varphi$'),
]

for data, cmap, out_path, add_quiver, cbar_label in colorbar_panels:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, origin='lower', cmap=cmap, extent=extent)
    if add_quiver:
        ax.quiver(X_clean[::step_clean, ::step_clean],
                  Y_clean[::step_clean, ::step_clean],
                  ux_zoom[::step_clean, ::step_clean],
                  uy_zoom[::step_clean, ::step_clean],
                  angles='xy', scale_units='xy',
                  color='white', width=0.004, alpha=0.7,
                  headwidth=0, headlength=0, headaxislength=0, pivot='mid')
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    fig.subplots_adjust(left=0, right=0.95, bottom=0, top=0.95)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Saved plot to {out_path}")