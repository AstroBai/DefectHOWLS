from tqdm import tqdm
import numpy as np
import os

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

data_om02 = np.load(f"{OM02_DIR}/OM02_stats_cov_mean_s234_fromshear.npz")
data_om04 = np.load(f"{OM04_DIR}/OM04_stats_cov_mean_s234_fromshear.npz")
data_s807 = np.load(f"{S807_DIR}/S807_stats_cov_mean_s234_fromshear.npz")
data_s809 = np.load(f"{S809_DIR}/S809_stats_cov_mean_s234_fromshear.npz")
data_slics = np.load(f"{SLICS_DIR}/SLICS_stats_cov_mean_s234_fromshear_subfields.npz")

params = np.array([[0.2000, 0.842],
                  [0.4000, 0.842],
                  [0.3134, 0.707],
                  [0.3134, 0.977]])

kappa_derivs = []
kappa_pos_derivs = []
kappa_neg_derivs = []
nn_pos_same2_derivs = []
nn_neg_same2_derivs = []
nn_pos_opp2_derivs = []
nn_neg_opp2_derivs = []
pp_derivs = []
nn_derivs = []
pn_derivs = []

for seed in range(128):
    for i in range(2):
        if i == 0:
            delta_p = params[1,0] - params[0,0]
            data_plus = np.load(f"{OM04_DIR}/OM04_stats_s234_fromshear_{seed}.npz")
            data_minus = np.load(f"{OM02_DIR}/OM02_stats_s234_fromshear_{seed}.npz")
        else:
            delta_p = params[3,1] - params[2,1]
            data_plus = np.load(f"{S809_DIR}/S809_stats_s234_fromshear_{seed}.npz")
            data_minus = np.load(f"{S807_DIR}/S807_stats_s234_fromshear_{seed}.npz")

        kappa_deriv = (data_plus['kappa'] - data_minus['kappa']) / delta_p
        kappa_pos_deriv = (data_plus['kappa_pos_cum'] - data_minus['kappa_pos_cum']) / delta_p
        kappa_neg_deriv = (data_plus['kappa_neg_cum'] - data_minus['kappa_neg_cum']) / delta_p
        nn_pos_same2_deriv = (data_plus['nn_pos_same2_cum'] - data_minus['nn_pos_same2_cum']) / delta_p
        nn_pos_opp2_deriv = (data_plus['nn_pos_opp2_cum'] - data_minus['nn_pos_opp2_cum']) / delta_p
        nn_neg_same2_deriv = (data_plus['nn_neg_same2_cum'] - data_minus['nn_neg_same2_cum']) / delta_p
        nn_neg_opp2_deriv = (data_plus['nn_neg_opp2_cum'] - data_minus['nn_neg_opp2_cum']) / delta_p
        pp_deriv = (data_plus['pp'] - data_minus['pp']) / delta_p
        nn_deriv = (data_plus['nn'] - data_minus['nn']) / delta_p
        pn_deriv = (data_plus['pn'] - data_minus['pn']) / delta_p
    
        kappa_derivs.append(kappa_deriv)
        kappa_pos_derivs.append(kappa_pos_deriv)
        kappa_neg_derivs.append(kappa_neg_deriv)
        nn_pos_same2_derivs.append(nn_pos_same2_deriv)
        nn_pos_opp2_derivs.append(nn_pos_opp2_deriv)
        nn_neg_same2_derivs.append(nn_neg_same2_deriv)
        nn_neg_opp2_derivs.append(nn_neg_opp2_deriv)
        pp_derivs.append(pp_deriv)
        nn_derivs.append(nn_deriv)
        pn_derivs.append(pn_deriv)

kappa_derivs = np.array(kappa_derivs)
kappa_pos_derivs = np.array(kappa_pos_derivs)
kappa_neg_derivs = np.array(kappa_neg_derivs)
nn_pos_same2_derivs = np.array(nn_pos_same2_derivs)
nn_pos_opp2_derivs = np.array(nn_pos_opp2_derivs)
nn_neg_same2_derivs = np.array(nn_neg_same2_derivs)
nn_neg_opp2_derivs = np.array(nn_neg_opp2_derivs)
pp_derivs = np.array(pp_derivs)
nn_derivs = np.array(nn_derivs)
pn_derivs = np.array(pn_derivs)

# Reshape to separate Om and s8 derivatives
# Current shape: (256, n_bins) -> (128, 2, n_bins)
# We want mean/cov for the vector [deriv_Om, deriv_s8] of length 2*n_bins
def get_mean_cov_combined(derivs_array):
    n_bins = derivs_array.shape[1]
    # Reshape to (seeds, params, bins)
    reshaped = derivs_array.reshape(128, 2, n_bins)
    # Flatten params and bins: (seeds, params*bins) -> [Om_bin1...Om_binN, s8_bin1...s8_binN]
    combined = reshaped.reshape(128, -1)
    mean = np.mean(combined, axis=0)
    cov = np.cov(combined, rowvar=False)
    return mean, cov, n_bins

kappa_derivs_mean, kappa_derivs_cov, n_bins_kappa = get_mean_cov_combined(kappa_derivs)
kappa_pos_derivs_mean, kappa_pos_derivs_cov, n_bins_pos = get_mean_cov_combined(kappa_pos_derivs)
kappa_neg_derivs_mean, kappa_neg_derivs_cov, n_bins_neg = get_mean_cov_combined(kappa_neg_derivs)
nn_pos_same2_derivs_mean, nn_pos_same2_derivs_cov, n_bins_nn = get_mean_cov_combined(nn_pos_same2_derivs)
nn_pos_opp2_derivs_mean, nn_pos_opp2_derivs_cov, _ = get_mean_cov_combined(nn_pos_opp2_derivs)
nn_neg_same2_derivs_mean, nn_neg_same2_derivs_cov, _ = get_mean_cov_combined(nn_neg_same2_derivs)
nn_neg_opp2_derivs_mean, nn_neg_opp2_derivs_cov, _ = get_mean_cov_combined(nn_neg_opp2_derivs)
pp_derivs_mean, pp_derivs_cov, n_bins_pp = get_mean_cov_combined(pp_derivs)
nn_derivs_mean, nn_derivs_cov, n_bins_nn2 = get_mean_cov_combined(nn_derivs)
pn_derivs_mean, pn_derivs_cov, n_bins_pn = get_mean_cov_combined(pn_derivs)

kappa_fid = data_slics['kappa_mean']
kappa_pos_fid = data_slics['kappa_pos_mean']
kappa_neg_fid = data_slics['kappa_neg_mean']

nn_pos_same1_fid = data_slics['nn_pos_same1_mean']
nn_pos_same2_fid = data_slics['nn_pos_same2_mean']
nn_pos_same4_fid = data_slics['nn_pos_same4_mean']
nn_pos_same8_fid = data_slics['nn_pos_same8_mean']

nn_pos_opp1_fid = data_slics['nn_pos_opp1_mean']
nn_pos_opp2_fid = data_slics['nn_pos_opp2_mean']
nn_pos_opp4_fid = data_slics['nn_pos_opp4_mean']
nn_pos_opp8_fid = data_slics['nn_pos_opp8_mean']

nn_neg_same1_fid = data_slics['nn_neg_same1_mean']
nn_neg_same2_fid = data_slics['nn_neg_same2_mean']
nn_neg_same4_fid = data_slics['nn_neg_same4_mean']
nn_neg_same8_fid = data_slics['nn_neg_same8_mean']

nn_neg_opp1_fid = data_slics['nn_neg_opp1_mean']
nn_neg_opp2_fid = data_slics['nn_neg_opp2_mean']
nn_neg_opp4_fid = data_slics['nn_neg_opp4_mean']
nn_neg_opp8_fid = data_slics['nn_neg_opp8_mean']


kappa_cov = data_slics['kappa_cov']
kappa_pos_cov = data_slics['kappa_pos_cov']
kappa_neg_cov = data_slics['kappa_neg_cov']
nn_pos_same2_cov = data_slics['nn_pos_same2_cov']
nn_pos_opp2_cov = data_slics['nn_pos_opp2_cov']

bin_edges = data_slics['bin_edges']
bin_edges1 = data_slics['bin_edges1']
bin_edges2 = data_slics['bin_edges2']
bin_edges4 = data_slics['bin_edges4']
bin_edges8 = data_slics['bin_edges8']
theta_bins = data_slics['theta']

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.formatter.use_mathtext'] = True


def sci_format_func(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exponent
    return r"${:.0f} \times 10^{{{}}}$".format(coeff, exponent)


if not os.path.exists('outfiles'):
    os.makedirs('outfiles')

N_samples = 128
# stride = 10


# 2. DPDF (Pos and Neg) - Combined
fig, ax = plt.subplots(1, figsize=(3, 2))


# Omega_m
y = kappa_pos_derivs_mean[0:n_bins_pos]
y_err = np.sqrt(np.diag(kappa_pos_derivs_cov[0:n_bins_pos, 0:n_bins_pos]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges, y, c='C0', ls='-', label=r'$\partial C^{\rm P}(\kappa) / \partial \Omega_{\rm m}$')
ax.fill_between(bin_edges, y - y_err, y + y_err, color='C0', alpha=0.1)
ax.fill_between(bin_edges, y - y_mean_error, y + y_mean_error, color='C0', alpha=0.3)

# Sigma_8
y = kappa_pos_derivs_mean[n_bins_pos:2*n_bins_pos]
y_err = np.sqrt(np.diag(kappa_pos_derivs_cov[n_bins_pos:2*n_bins_pos, n_bins_pos:2*n_bins_pos]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges, y, c='C1', ls='-', label=r'$\partial C^{\rm P}(\kappa) / \partial \sigma_8$')
ax.fill_between(bin_edges, y - y_err, y + y_err, color='C1', alpha=0.1)
ax.fill_between(bin_edges, y - y_mean_error, y + y_mean_error, color='C1', alpha=0.3)

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$\partial {\rm CDF}(\kappa) / \partial p$')
# ax.set_title(r'$\partial C^{\rm P}(\kappa) / \partial p$')
ax.set_xticks(np.array([-0.04, -0.02, 0, 0.02, 0.04]))

# Neg
# Omega_m
y = kappa_neg_derivs_mean[0:n_bins_neg]
y_err = np.sqrt(np.diag(kappa_neg_derivs_cov[0:n_bins_neg, 0:n_bins_neg]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges, y, c='C0', ls='--', label=r'$\partial C^{\rm N}(\kappa) / \partial \Omega_{\rm m}$')


# Sigma_8
y = kappa_neg_derivs_mean[n_bins_neg:2*n_bins_neg]
y_err = np.sqrt(np.diag(kappa_neg_derivs_cov[n_bins_neg:2*n_bins_neg, n_bins_neg:2*n_bins_neg]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges, y, c='C1', ls='--', label=r'$\partial C^{\rm N}(\kappa) / \partial \sigma_8$')

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\kappa$')
# ax.set_title(r'$\partial C^{\rm N}(\kappa) / \partial p$')
ax.set_xticks(np.array([-0.04, -0.02, 0, 0.02, 0.04]))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_format_func))

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))

plt.subplots_adjust(wspace=0)
plt.savefig('outfiles/derivs_dcdf.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. NNs
fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
n = n_bins_nn

# 2NN Same
ax = axes[0]
ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_format_func))
# Omega_m
y = nn_pos_same2_derivs_mean[0:n]
y_err = np.sqrt(np.diag(nn_pos_same2_derivs_cov[0:n, 0:n]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges2, y, c='C0', ls='-', label=r'$\partial C^{\rm PP}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')
ax.fill_between(bin_edges2, y - y_err, y + y_err, color='C0', alpha=0.1)
ax.fill_between(bin_edges2, y - y_mean_error, y + y_mean_error, color='C0', alpha=0.3)
y_prime = nn_neg_same2_derivs_mean[0:n]
ax.plot(bin_edges2, y_prime, c='C0', ls='--', label=r'$\partial C^{\rm NN}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')
# Sigma_8
y = nn_pos_same2_derivs_mean[n:2*n]
y_err = np.sqrt(np.diag(nn_pos_same2_derivs_cov[n:2*n, n:2*n]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges2, y, c='C1', ls='-', label=r'$\partial C^{\rm PP}_{\rm 2NN}(\theta) / \partial \sigma_8$')
ax.fill_between(bin_edges2, y - y_err, y + y_err, color='C1', alpha=0.1)
ax.fill_between(bin_edges2, y - y_mean_error, y + y_mean_error, color='C1', alpha=0.3)

y_prime = nn_neg_same2_derivs_mean[n:2*n]
ax.plot(bin_edges2, y_prime, c='C1', ls='--', label=r'$\partial C^{\rm NN}_{\rm 2NN}(\theta) / \partial \sigma_8$')

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# ax.set_title(r'$\partial C^{\rm PP}(\theta) / \partial p$')
ax.set_ylabel(r'$\partial {\rm CDF}(\theta) / \partial p$')
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_xscale('log')
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=6)

# 2NN Opp
ax = axes[1]
# Omega_m
y = nn_pos_opp2_derivs_mean[0:n]
y_err = np.sqrt(np.diag(nn_pos_opp2_derivs_cov[0:n, 0:n]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges2, y, c='C0', ls='-', label=r'$\partial C^{\rm PN}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')
ax.fill_between(bin_edges2, y - y_err, y + y_err, color='C0', alpha=0.1)
ax.fill_between(bin_edges2, y - y_mean_error, y + y_mean_error, color='C0', alpha=0.3)
y_prime = nn_neg_opp2_derivs_mean[0:n]
ax.plot(bin_edges2, y_prime, c='C0', ls='--', label=r'$\partial C^{\rm NP}_{\rm 2NN}(\theta) / \partial \Omega_{\rm m}$')

# Sigma_8
y = nn_pos_opp2_derivs_mean[n:2*n]
y_err = np.sqrt(np.diag(nn_pos_opp2_derivs_cov[n:2*n, n:2*n]))
y_mean_error = y_err / np.sqrt(N_samples)
ax.plot(bin_edges2, y, c='C1', ls='-', label=r'$\partial C^{\rm PN}_{\rm 2NN}(\theta) / \partial \sigma_8$')
ax.fill_between(bin_edges2, y - y_err, y + y_err, color='C1', alpha=0.1)
ax.fill_between(bin_edges2, y - y_mean_error, y + y_mean_error, color='C1', alpha=0.3)
y_prime = nn_neg_opp2_derivs_mean[n:2*n]
ax.plot(bin_edges2, y_prime, c='C1', ls='--', label=r'$\partial C^{\rm NP}_{\rm 2NN}(\theta) / \partial \sigma_8$')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# ax.set_title(r'$\partial C^{\rm PN}(\theta) / \partial p$')
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_xscale('log')
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), fontsize=6)

# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))
plt.subplots_adjust(wspace=0)
plt.savefig('outfiles/derivs_nns.png', dpi=300, bbox_inches='tight')
plt.close()


#====================================================================================
color_map = plt.get_cmap('tab10')
# compare positive and negative CDF, and rescaled overall pdf (cumulative)
kappa_overall_cdf = np.cumsum(kappa_fid)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])    
plt.figure(figsize=(4, 3))
plt.plot(bin_edges, kappa_pos_fid/kappa_pos_fid[-1], label='DCDF+', c=color_map(0))
plt.plot(bin_edges, kappa_neg_fid/kappa_neg_fid[-1], label='DCDF-', c=color_map(1))
plt.plot(bin_centers, kappa_overall_cdf/kappa_overall_cdf[-1], label='CDF, All Pixels', c='grey', linestyle='dashed')
plt.xlabel(r'$\kappa$')
plt.ylabel('Cumulative Number Distribution')
plt.title('Comparison of CDFs')
plt.legend()
# plt.grid(True, alpha=0.1)
plt.savefig('outfiles/compare_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# omegam+ cdf, omegam- cdf, fid cdf | sigma8+ cdf, sigma8- cdf, fid cdf
kappa_pos_om02 = data_om02['kappa_pos_mean']
kappa_pos_om04 = data_om04['kappa_pos_mean']
kappa_neg_om02 = data_om02['kappa_neg_mean']
kappa_neg_om04 = data_om04['kappa_neg_mean']
kappa_pos_s807 = data_s807['kappa_pos_mean']
kappa_pos_s809 = data_s809['kappa_pos_mean']
kappa_neg_s807 = data_s807['kappa_neg_mean']
kappa_neg_s809 = data_s809['kappa_neg_mean']
plt.figure(figsize=(4, 3))
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
# Omegam
ax = axes[0]
ax.plot(bin_edges, kappa_pos_fid, label='Fiducial DCDF+', c='black')
ax.plot(bin_edges, kappa_neg_fid, label='Fiducial DCDF-', c='grey')
ax.plot(bin_edges, kappa_pos_om04, label=r'$\Omega_m+$ DCDF+', c=color_map(0))
ax.plot(bin_edges, kappa_pos_om02, label=r'$\Omega_m-$ DCDF+', c=color_map(0), linestyle='dashed')
ax.plot(bin_edges, kappa_neg_om04, label=r'$\Omega_m+$ DCDF-', c=color_map(1))
ax.plot(bin_edges, kappa_neg_om02, label=r'$\Omega_m-$ DCDF-', c=color_map(1), linestyle='dashed')
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$C^{p,n}(\kappa)$')
ax.set_title(r'Effect of $\Omega_m$ on DCDFs')
# Sigma8
ax = axes[1]
ax.plot(bin_edges, kappa_pos_fid, label='Fiducial DCDF+', c='black')
ax.plot(bin_edges, kappa_neg_fid, label='Fiducial DCDF-', c='grey')
ax.plot(bin_edges, kappa_pos_s809, label=r'$\sigma_8+$ DCDF+', c=color_map(0))
ax.plot(bin_edges, kappa_pos_s807, label=r'$\sigma_8-$ DCDF+', c=color_map(0), linestyle='dashed')
ax.plot(bin_edges, kappa_neg_s809, label=r'$\sigma_8+$ DCDF-', c=color_map(1))
ax.plot(bin_edges, kappa_neg_s807, label=r'$\sigma_8-$ DCDF-', c=color_map(1), linestyle='dashed')
ax.set_xlabel(r'$\kappa$')
ax.set_title(r'Effect of $\sigma_8$ on DCDFs')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 0.55))
plt.subplots_adjust(wspace=0)
# plt.grid(True, alpha=0.1)
plt.savefig('outfiles/sg8_om_cdf.png', dpi=300, bbox_inches='tight')
plt.close() 

# dpdf, pdf cov

all_and_pdf_cov = data_slics['all_and_pdf_cov']
# take first 60 and last 28
cdf_cov = all_and_pdf_cov[np.ix_(list(range(60)) + list(range(all_and_pdf_cov.shape[0]-28, all_and_pdf_cov.shape[0])), list(range(60)) + list(range(all_and_pdf_cov.shape[0]-28, all_and_pdf_cov.shape[0])))]

plt.figure(figsize=(4, 3))
# normalize covariance to correlation matrix
d = np.sqrt(np.diag(cdf_cov))
corr = cdf_cov / np.outer(d, d)
plt.imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Matrix (DCDF + PDF)')
# add lines to separate different parts
plt.axvline(x=29.5, color='black', linestyle='--', linewidth=1)
plt.axhline(y=29.5, color='black', linestyle='--', linewidth=1)
plt.axvline(x=60.5, color='black', linestyle='--', linewidth=1)
plt.axhline(y=60.5, color='black', linestyle='--', linewidth=1)

# ticks
plt.xticks([15, 45, 74], [r'$DCDF+$', r'$DCDF-$', r'PDF'])
plt.yticks([15, 45, 74], [r'$DCDF+$', r'$DCDF-$', r'PDF'])
plt.tick_params(axis='both', which='both', length=0)
plt.savefig('outfiles/corr_matrix_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# compare 1, 2, 4, 8 NN
plt.figure(figsize=(4, 3))
plt.plot(bin_edges1, nn_pos_same1_fid, label='1NN ++', c=color_map(0))
plt.plot(bin_edges1, nn_pos_opp1_fid, label='1NN +-', c=color_map(0), linestyle='dashed')
plt.plot(bin_edges2, nn_pos_same2_fid, label='2NN ++', c=color_map(1))
plt.plot(bin_edges2, nn_pos_opp2_fid, label='2NN +-', c=color_map(1), linestyle='dashed')
plt.plot(bin_edges4, nn_pos_same4_fid, label='4NN ++', c=color_map(2))
plt.plot(bin_edges4, nn_pos_opp4_fid, label='4NN +-', c=color_map(2), linestyle='dashed')
plt.plot(bin_edges8, nn_pos_same8_fid, label='8NN ++', c=color_map(3))
plt.plot(bin_edges8, nn_pos_opp8_fid, label='8NN +-', c=color_map(3), linestyle='dashed')
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Distance (arcmin)')
plt.ylabel('Cumulative Number Distribution')
plt.title('Comparison of kNNs')
plt.legend()
# plt.grid(True, alpha=0.1)
plt.savefig('outfiles/compare_knn_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# compare pos 2nn ++/+- and neg 2nn --/-+
plt.figure(figsize=(4.8, 3))
plt.plot(bin_edges2, nn_pos_same2_fid, label='2NN ++', c=color_map(0))
plt.plot(bin_edges2, nn_pos_opp2_fid, label='2NN +-', c=color_map(0), linestyle='dashed')
plt.plot(bin_edges2, nn_neg_same2_fid, label='2NN --', c=color_map(1))
plt.plot(bin_edges2, nn_neg_opp2_fid, label='2NN -+', c=color_map(1), linestyle='dashed')
plt.xlabel('Distance (arcmin)')
plt.ylabel('Cumulative Number Distribution')
# plt.title('Comparison of 2NN Positive and Negative')
plt.legend(loc='upper left')
plt.text(0.9, 0.1, 'Euclid-like', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12)
# plt.grid(True, alpha=0.1)
plt.savefig('outfiles/compare_2nn_pos_neg_cdf.png', dpi=300)
plt.close()

print("total number density derivative for all defects (Om):", kappa_pos_derivs_mean[n_bins_pos-1]+kappa_neg_derivs_mean[n_bins_neg-1])
print("total number density derivative for all defects (s8):", kappa_pos_derivs_mean[2*n_bins_pos-1]+kappa_neg_derivs_mean[2*n_bins_neg-1])