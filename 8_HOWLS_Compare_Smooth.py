import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import plots
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8

# use tab10 colors
color_map = plt.get_cmap('tab10')

fisher_s234_gaussian_fromshear = np.load('outfiles/defects_fisher_s234_gaussian_fromshear.npy')
fisher_s234_mexican_fromshear = np.load('outfiles/defects_fisher_s234_mexican_fromshear.npy')
fisher_s467_th_fromshear = np.load('outfiles/defects_fisher_s467_th_fromshear.npy')
fisher_s467_th = np.load('outfiles/defects_fisher_s467_th.npy')

covmat_s234_gaussian_fromshear = np.linalg.inv(fisher_s234_gaussian_fromshear)
covmat_s234_mexican_fromshear = np.linalg.inv(fisher_s234_mexican_fromshear)
covmat_s467_th_fromshear = np.linalg.inv(fisher_s467_th_fromshear)
covmat_s467_th = np.linalg.inv(fisher_s467_th)
# add S8 to the covariance matrices
params_fid = np.array([0.3134, 0.842])
def add_s8_to_cov(covmat, params_fid):
    Om_fid, sigma8_fid = params_fid
    s8_fid = sigma8_fid * (Om_fid / 0.3)**0.5
    
    # Create a new covariance matrix with an additional row and column for S8
    new_cov = np.zeros((3, 3))
    new_cov[:2, :2] = covmat
    
    # Calculate the variance of S8 using error propagation
    var_s8 = (s8_fid**2) * ((covmat[0, 0] / Om_fid**2) + (covmat[1, 1] / sigma8_fid**2) - (2 * covmat[0, 1] / (Om_fid * sigma8_fid)))
    
    new_cov[2, 2] = var_s8
    
    return new_cov

covmat_s234_gaussian_fromshear = add_s8_to_cov(covmat_s234_gaussian_fromshear, params_fid)
covmat_s234_mexican_fromshear = add_s8_to_cov(covmat_s234_mexican_fromshear, params_fid)
covmat_s467_th_fromshear = add_s8_to_cov(covmat_s467_th_fromshear, params_fid)
covmat_s467_th = add_s8_to_cov(covmat_s467_th, params_fid)
params_fid = np.append(params_fid, params_fid[1] * (params_fid[0] / 0.3)**0.5) # add S8 to the fiducial parameters
names = ['Om', 'sigma8', 'S8']
labels = ['\\Omega_{\\rm m}', '\\sigma_{\\rm 8}', 'S_8']
s234_gaussian_fromshear_samples = GaussianND(mean=params_fid, cov=covmat_s234_gaussian_fromshear, names=names, labels=labels)
s234_mexican_fromshear_samples = GaussianND(mean=params_fid, cov=covmat_s234_mexican_fromshear, names=names, labels=labels)
s467_th_fromshear_samples = GaussianND(mean=params_fid, cov=covmat_s467_th_fromshear, names=names, labels=labels)
s467_th_samples = GaussianND(mean=params_fid, cov=covmat_s467_th, names=names, labels=labels)

g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=1)
g.settings.num_plot_contours = 1
g.triangle_plot([s234_gaussian_fromshear_samples, s234_mexican_fromshear_samples, s467_th_fromshear_samples, s467_th_samples],
                filled=False, contour_colors=[color_map(0), color_map(1), color_map(2), color_map(3)], contour_ls=['-', '-', '-', '-'], contour_lw=1.5, legend_labels=['Top-hat on Convergence (4.69\')', 'Top-hat on Shear (4.69\')', 'Gaussian on Shear (2.34\')', 'Mexican-hat on Shear (2.34\')'], legend_loc='upper right', fontsize=7)
# g.add_legend(legend_labels=['Top-hat on Convergence (4.69\')', 'Top-hat on Shear (4.69\')', 'Gaussian on Shear (2.34\')', 'Mexican-hat on Shear (2.34\')'], legend_loc='upper right',fontsize=7)
plt.savefig('outfiles/Defects_Compare_Smooth.png', dpi=300)
plt.close()
