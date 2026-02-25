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
g.triangle_plot([s467_th_samples, s467_th_fromshear_samples, s234_gaussian_fromshear_samples, s234_mexican_fromshear_samples],
                filled=False, contour_colors=[color_map(0), color_map(1), color_map(2), color_map(3)], contour_ls=['-', '-', '-', '-'], contour_lw=1.5, legend_labels=['Top-hat on Convergence (4.69\')', 'Top-hat on Shear (4.69\')', 'Gaussian on Shear (2.34\')', 'Mexican-hat on Shear (2.34\')'], legend_loc='upper right', fontsize=7)
# g.add_legend(legend_labels=['Top-hat on Convergence (4.69\')', 'Top-hat on Shear (4.69\')', 'Gaussian on Shear (2.34\')', 'Mexican-hat on Shear (2.34\')'], legend_loc='upper right',fontsize=7)
plt.savefig('outfiles/Defects_Compare_Smooth.png', dpi=300)
plt.close()


# compare with Euclid HOWLS results
params_fid = np.array([0.3134, 0.842])
fisher_pdf_s467_th = np.load('outfiles/pdf_fisher_s467_th_lessbins_KL0.1.npy')


inv_fisher_pdf_ours = np.linalg.inv(fisher_pdf_s467_th) 
inv_fisher_pdf_euclid = np.array([  [4.81638062E-06,  -7.51896823E-06],
                                    [-7.51887683E-06,   1.25822144E-05]])
inv_fisher_defects = np.linalg.inv(fisher_s467_th)
# one_sigma_pdf_theory = np.array([[0.31010797872340423, 0.8473276595744681],
#                                  [0.31669840425531914, 0.836604255319149],
#                                  [0.3109533244680851, 0.8467148936170212],
#                                  [0.3118469414893617, 0.8455659574468085],
#                                  [0.3125997340425532, 0.8444936170212766],
#                                  [0.31330565159574464, 0.8434212765957446],
#                                  [0.3141059175531915, 0.8421191489361702],
#                                  [0.3154716755319149, 0.8396680851063829],
#                                  [0.31631981382978724, 0.8379829787234042],
#                                  [0.3153829122340425, 0.8377531914893617],
#                                  [0.3104392952127659, 0.8461021276595745],
#                                  [0.3111934840425532, 0.8444936170212766],
#                                  [0.3159940824468085, 0.837063829787234],
#                                  [0.3119472739361702, 0.8430382978723404]])
# inv_fisher_theory = np.cov(one_sigma_pdf_theory, rowvar=False)
# print(np.mean(one_sigma_pdf_theory, axis=0))

covmat_pdf = add_s8_to_cov(inv_fisher_pdf_ours, params_fid) 
covmat_pdf_euclid = add_s8_to_cov(inv_fisher_pdf_euclid, params_fid)
covmat_defects = add_s8_to_cov(inv_fisher_defects, params_fid) 
# covmat_pdf_theory = add_s8_to_cov(inv_fisher_theory, params_fid)
params_fid = np.append(params_fid, params_fid[1] * (params_fid[0] / 0.3)**0.5) # add S8 to the fiducial parameters

pdf_samples = GaussianND(mean=params_fid, cov=covmat_pdf, names=names, labels=labels)
pdf_euclid_samples = GaussianND(mean=params_fid, cov=covmat_pdf_euclid, names=names, labels=labels)
# pdf_theory_samples = GaussianND(mean=params_fid, cov=covmat_pdf_theory, names=names, labels=labels)
defects_samples = GaussianND(mean=params_fid, cov=covmat_defects, names=names, labels=labels)


g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=1)
g.settings.num_plot_contours = 1
# linewith = 1.5
g.settings.linewidth_contour = 1.5
g.settings.linewidth = 1.5
g.triangle_plot([defects_samples, pdf_samples, pdf_euclid_samples],
                filled=False, contour_colors=[color_map(0), color_map(1), color_map(2)], contour_ls=['-', '-', '-'], contour_lw=2, legend_labels=['Top-hat on Convergence (4.69\')', 'PDF on Shear (4.69\')', 'PDF on Shear (Euclid)'], legend_loc='upper right', fontsize=7)
plt.savefig('outfiles/Defects_Compare_PDF.png', dpi=300)
plt.close()

# print 1 sigma/parameter_fid
print("Defects 1-sigma:", np.sqrt(np.diag(covmat_defects)) / params_fid)
print("PDF 1-sigma:", np.sqrt(np.diag(covmat_pdf)) / params_fid)
print("Euclid PDF 1-sigma:", np.sqrt(np.diag(covmat_pdf_euclid)) / params_fid)