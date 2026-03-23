import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import os
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
params_fid = np.array([0.3134, 0.842])
names = ['Om', 'sigma8']
labels = ['\\Omega_{\\rm m}', '\\sigma_{\\rm 8}']
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
fisher_pdf_s467_th = np.load('outfiles/pdf_fisher_s467_th_KL0.1_nosub.npy')


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

covmat_pdf = inv_fisher_pdf_ours
covmat_pdf_euclid = inv_fisher_pdf_euclid
covmat_defects = inv_fisher_defects
# covmat_pdf_theory = inv_fisher_theory

pdf_samples = GaussianND(mean=params_fid, cov=covmat_pdf, names=names, labels=labels)
pdf_euclid_samples = GaussianND(mean=params_fid, cov=covmat_pdf_euclid, names=names, labels=labels)
# pdf_theory_samples = GaussianND(mean=params_fid, cov=covmat_pdf_theory, names=names, labels=labels)
defects_samples = GaussianND(mean=params_fid, cov=covmat_defects, names=names, labels=labels)


g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=1)
g.settings.num_plot_contours = 1
# linewith = 1.5
g.settings.linewidth_contour = 1.5
g.settings.linewidth = 1.5
g.triangle_plot([pdf_samples, pdf_euclid_samples],
                filled=False, contour_colors=[color_map(0), color_map(1)], contour_ls=['-', '-'], contour_lw=2, legend_labels=['PDF on Shear (4.69\')', 'PDF on Shear (Euclid)'], legend_loc='upper right', fontsize=7)
plt.savefig('outfiles/Defects_Compare_PDF.png', dpi=300)
plt.close()

# print 1 sigma/parameter_fid
print("Defects 1-sigma:", np.sqrt(np.diag(covmat_defects)) / params_fid)
print("PDF 1-sigma:", np.sqrt(np.diag(covmat_pdf)) / params_fid)
print("Euclid PDF 1-sigma:", np.sqrt(np.diag(covmat_pdf_euclid)) / params_fid)


# # defect_lessbins vs more bins
# defect_lessbins = np.load('outfiles/defects_fisher_s467_th_lessbins.npy')
# defect_morebins = np.load('outfiles/defects_fisher_s467_th_morebins.npy')

# covmat_defect_lessbins = np.linalg.inv(defect_lessbins)
# covmat_defect_morebins = np.linalg.inv(defect_morebins)

# # don't add S8
# params_fid_lessbins = np.array([0.3134, 0.842])
# params_fid_morebins = np.array([0.3134, 0.842])

# defect_lessbins_samples = GaussianND(mean=params_fid_lessbins, cov=covmat_defect_lessbins, names=names[:2], labels=labels[:2])
# defect_morebins_samples = GaussianND(mean=params_fid_morebins, cov=covmat_defect_morebins, names=names[:2], labels=labels[:2])  
# g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=1)
# g.settings.num_plot_contours = 1
# g.settings.linewidth_contour = 1.5
# g.settings.linewidth = 1.5
# g.triangle_plot([defect_lessbins_samples, defect_morebins_samples],
#                 filled=False, contour_colors=[color_map(0), color_map(1)], contour_ls=['-', '-'], contour_lw=2, legend_labels=['Defects (less bins)', 'Defects (more bins)'], legend_loc='upper right', fontsize=7)
# plt.savefig('outfiles/Defects_Compare_Bins.png', dpi=300)
# plt.close()


# # compare pdf with more bins vs less bins
# pdf_lessbins = np.load('outfiles/pdf_fisher_s467_th_lessbins_KL0.1.npy')
# pdf_morebins = np.load('outfiles/pdf_fisher_s467_th_morebins_KL0.1.npy')
# covmat_pdf_lessbins = np.linalg.inv(pdf_lessbins)
# covmat_pdf_morebins = np.linalg.inv(pdf_morebins)
# pdf_lessbins_samples = GaussianND(mean=params_fid_lessbins, cov=covmat_pdf_lessbins, names=names[:2], labels=labels[:2])
# pdf_morebins_samples = GaussianND(mean=params_fid_morebins, cov=covmat_pdf_morebins, names=names[:2], labels=labels[:2])
# g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=1)
# g.settings.num_plot_contours = 1
# g.settings.linewidth_contour = 1.5
# g.settings.linewidth = 1.5
# g.triangle_plot([pdf_lessbins_samples, pdf_morebins_samples],
#                 filled=False, contour_colors=[color_map(0), color_map(1)], contour_ls=['-', '-'], contour_lw=2, legend_labels=['PDF (less bins)', 'PDF (more bins)'], legend_loc='upper right', fontsize=7)
# plt.savefig('outfiles/PDF_Compare_Bins.png', dpi=300)
# plt.close()

# compare our kappa 2pcf fisher with Euclid's kappa 2pcf fisher
params_fid = np.array([0.3134, 0.842])
fisher_kk_s467_th = np.load('outfiles/kk_fisher_s467_th_KL0.1_nosub.npy')
inv_fisher_kk_ours = np.linalg.inv(fisher_kk_s467_th)
inv_fisher_kk_euclid = np.array([[   3.48024769E-05,  -5.54870130E-05],
                                 [-5.54869766E-05,   8.99243387E-05]])
covmat_kk_ours = inv_fisher_kk_ours
covmat_kk_euclid = inv_fisher_kk_euclid
kk_ours_samples = GaussianND(mean=params_fid, cov=covmat_kk_ours, names=names, labels=labels)
kk_euclid_samples = GaussianND(mean=params_fid, cov=covmat_kk_euclid, names=names, labels=labels)
g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=1)
g.settings.num_plot_contours = 1
g.settings.linewidth_contour = 1.5
g.settings.linewidth = 1.5
g.triangle_plot([kk_ours_samples, kk_euclid_samples],
                filled=False, contour_colors=[color_map(0), color_map(1)], contour_ls=['-', '-'], contour_lw=2, legend_labels=[r'$\kappa$-2PCF (4.69\')', r'$\kappa$-2PCF (Euclid)'], legend_loc='upper right', fontsize=7)
plt.savefig('outfiles/Defects_Compare_KK.png', dpi=300)
plt.close()


def _load_inv_fisher(candidates):
    for path in candidates:
        if os.path.exists(path):
            return np.linalg.inv(np.load(path)), path
    raise FileNotFoundError(f"None of these files exist: {candidates}")


# -----------------------------------------------------------------
# Horizontal errorbar-like comparison (PDF/Defect KL0.1, nosub, Euclid)
# -----------------------------------------------------------------
params_fid = np.array([0.3134, 0.842])
fid_om = params_fid[0]
fid_s8 = params_fid[1]

cov_pdf_kl01, file_pdf_kl01 = _load_inv_fisher([
    'outfiles/pdf_fisher_s467_th_KL0.1.npy',
])
cov_pdf_kl01_nosub, file_pdf_kl01_nosub = _load_inv_fisher([
    'outfiles/pdf_fisher_s467_th_KL0.1_nosub.npy',
])
cov_defect_kl01, file_defect_kl01 = _load_inv_fisher([
    'outfiles/defects_fisher_s467_th.npy',
])
cov_defect_kl01_nosub, file_defect_kl01_nosub = _load_inv_fisher([
    'outfiles/defects_fisher_s467_th_nosub.npy',
])
cov_kk_kl01_nosub, file_kk_kl01_nosub = _load_inv_fisher([
    'outfiles/kk_fisher_s467_th_KL0.1_nosub.npy',
])

cov_kk_euclid = inv_fisher_kk_euclid
cov_pdf_euclid = inv_fisher_pdf_euclid

print('Using files for bar-style comparison:')
print('  PDF KL0.1:', file_pdf_kl01)
print('  PDF KL0.1 nosub:', file_pdf_kl01_nosub)
print('  Defect KL0.1:', file_defect_kl01)
print('  Defect KL0.1 nosub:', file_defect_kl01_nosub)
print('  KK KL0.1 nosub:', file_kk_kl01_nosub)
print('  PDF Euclid:', 'from code')
print('  KK Euclid:', 'from code')


bar_data = [
    ('PDF th KL0.1', cov_pdf_kl01),
    ('PDF th KL0.1 nosub', cov_pdf_kl01_nosub),
    ('PDF Euclid', cov_pdf_euclid),
    ('Defect th KL0.1', cov_defect_kl01),
    ('Defect th KL0.1 nosub', cov_defect_kl01_nosub),
    ('KK th KL0.1 nosub', cov_kk_kl01_nosub),
    ('KK Euclid', cov_kk_euclid),
]

labels = []
vals_om = []
vals_s8 = []
for label, cov in bar_data:
    labels.append(label)
    vals_om.append(np.sqrt(cov[0, 0]) / fid_om * 100.0)
    vals_s8.append(np.sqrt(cov[1, 1]) / fid_s8 * 100.0)

vals_om = np.array(vals_om)
vals_s8 = np.array(vals_s8)
y_pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8, 3.6))
min_om, max_om = vals_om.min(), vals_om.max()
min_s8, max_s8 = vals_s8.min(), vals_s8.max()

for i, (om, s8) in enumerate(zip(vals_om, vals_s8)):
    if max_om > min_om:
        alpha_om = 0.8 - 0.6 * (om - min_om) / (max_om - min_om)
    else:
        alpha_om = 0.8
    if max_s8 > min_s8:
        alpha_s8 = 0.8 - 0.6 * (s8 - min_s8) / (max_s8 - min_s8)
    else:
        alpha_s8 = 0.8

    color_om = cm.Blues(0.3 + 0.7 * alpha_om)
    color_s8 = cm.Oranges(0.3 + 0.7 * alpha_s8)

    marker = '*' if labels[i].startswith('Defect') else 's'
    msize = 9 if marker == '*' else 6

    ax.plot([0, -om], [i, i], color=color_om, linewidth=2, marker=marker, markevery=[1], markersize=msize, markerfacecolor=color_om)
    ax.plot([0, s8], [i, i], color=color_s8, linewidth=2, marker=marker, markevery=[1], markersize=msize, markerfacecolor=color_s8)

    ax.text(-om - 0.06, i, f'{om:.2f}%', va='center', ha='right', fontsize=8, color=color_om)
    ax.text(s8 + 0.06, i, f'{s8:.2f}%', va='center', ha='left', fontsize=8, color=color_s8)

ax.plot([], [], color=cm.Blues(0.8), linewidth=2, marker='s', markersize=6, label=r'$\delta \Omega_{\rm m}/\Omega_{\rm m}$ [%]')
ax.plot([], [], color=cm.Oranges(0.8), linewidth=2, marker='s', markersize=6, label=r'$\delta \sigma_8/\sigma_8$ [%]')

ax.axhline(y=2.5, color='black', linestyle='--', linewidth=1)
ax.text(-max(vals_om) * 1.35, 1.2, 'PDF', va='center', ha='left', fontsize=9)
ax.text(-max(vals_om) * 1.35, 3.8, 'Defect', va='center', ha='left', fontsize=9)

ax.axvspan(-vals_om[-1], vals_s8[-1], color='gray', alpha=0.2)
ax.axvline(x=0, color='black', linewidth=0.8)

ax.set_xlim(-max(vals_om) * 1.45, max(vals_s8) * 1.45)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ticks = ax.get_xticks()
ax.set_xticklabels([f'{abs(t):.1f}' for t in ticks])
ax.set_xlabel('Relative Precision [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

plt.tight_layout()
plt.savefig('outfiles/1sigma_bars_pdf_defect_KL0.1.png', dpi=300)
plt.close()