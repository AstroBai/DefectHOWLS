from tqdm import tqdm
import numpy as np
import os

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

data_om02 = np.load(f"{OM02_DIR}/OM02_stats_cov_mean_s467.npz")
data_om04 = np.load(f"{OM04_DIR}/OM04_stats_cov_mean_s467.npz")
data_s807 = np.load(f"{S807_DIR}/S807_stats_cov_mean_s467.npz")
data_s809 = np.load(f"{S809_DIR}/S809_stats_cov_mean_s467.npz")
data_slics = np.load(f"{SLICS_DIR}/SLICS_stats_cov_mean_s467.npz")

params = np.array([[0.2000, 0.842],
                  [0.4000, 0.842],
                  [0.3134, 0.707],
                  [0.3134, 0.977]])

kappa_derivs = []
kappa_pos_derivs = []
kappa_neg_derivs = []
nn_pos_same2_derivs = []
nn_pos_opp2_derivs = []

for seed in range(128):
    for i in range(2):
        if i == 0:
            delta_p = params[1,0] - params[0,0]
            data_plus = np.load(f"{OM04_DIR}/OM04_stats_s467_{seed}.npz")
            data_minus = np.load(f"{OM02_DIR}/OM02_stats_s467_{seed}.npz")
        else:
            delta_p = params[3,1] - params[2,1]
            data_plus = np.load(f"{S809_DIR}/S809_stats_s467_{seed}.npz")
            data_minus = np.load(f"{S807_DIR}/S807_stats_s467_{seed}.npz")

        kappa_deriv = (data_plus['kappa'] - data_minus['kappa']) / delta_p
        kappa_pos_deriv = (data_plus['kappa_pos_cum'] - data_minus['kappa_pos_cum']) / delta_p
        kappa_neg_deriv = (data_plus['kappa_neg_cum'] - data_minus['kappa_neg_cum']) / delta_p
        nn_pos_same2_deriv = (data_plus['nn_pos_same2_cum'] - data_minus['nn_pos_same2_cum']) / delta_p
        nn_pos_opp2_deriv = (data_plus['nn_pos_opp2_cum'] - data_minus['nn_pos_opp2_cum']) / delta_p
        kappa_derivs.append(kappa_deriv)
        kappa_pos_derivs.append(kappa_pos_deriv)
        kappa_neg_derivs.append(kappa_neg_deriv)
        nn_pos_same2_derivs.append(nn_pos_same2_deriv)
        nn_pos_opp2_derivs.append(nn_pos_opp2_deriv)

kappa_derivs = np.array(kappa_derivs)
kappa_pos_derivs = np.array(kappa_pos_derivs)
kappa_neg_derivs = np.array(kappa_neg_derivs)
nn_pos_same2_derivs = np.array(nn_pos_same2_derivs)
nn_pos_opp2_derivs = np.array(nn_pos_opp2_derivs)

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

kappa_fid = data_slics['kappa_mean']
kappa_pos_fid = data_slics['kappa_pos_mean']
kappa_neg_fid = data_slics['kappa_neg_mean']
nn_pos_same2_fid = data_slics['nn_pos_same2_mean']
nn_pos_opp2_fid = data_slics['nn_pos_opp2_mean']

kappa_cov = data_slics['kappa_cov']
kappa_pos_cov = data_slics['kappa_pos_cov']
kappa_neg_cov = data_slics['kappa_neg_cov']
nn_pos_same2_cov = data_slics['nn_pos_same2_cov']
nn_pos_opp2_cov = data_slics['nn_pos_opp2_cov']

bin_edges = data_slics['bin_edges']
bin_edges2 = data_slics['bin_edges2']

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

if not os.path.exists('outfiles'):
    os.makedirs('outfiles')

N_samples = 128

# 1. PDF
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
n = n_bins_kappa

ax.plot(bin_centers, kappa_fid, c='black')
ax.errorbar(bin_centers, kappa_fid, yerr=np.sqrt(np.diag(kappa_cov)) / np.sqrt(924), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_centers, kappa_derivs_mean[0:n], 
            yerr=np.sqrt(np.diag(kappa_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_centers, kappa_derivs_mean[n:2*n], 
            yerr=np.sqrt(np.diag(kappa_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_centers, kappa_derivs_mean[0:n] - np.sqrt(np.diag(kappa_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), kappa_derivs_mean[0:n] + np.sqrt(np.diag(kappa_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), alpha=0.3)
ax.fill_between(bin_centers, kappa_derivs_mean[n:2*n] - np.sqrt(np.diag(kappa_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), kappa_derivs_mean[n:2*n] + np.sqrt(np.diag(kappa_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), alpha=0.3)

ax.set_xlabel(r'$\kappa$')
ax.set_ylabel('Probability Density')
ax.grid(True, alpha=0.3)
plt.legend()
plt.savefig('outfiles/derivs_pdf.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. DPDF (Pos and Neg)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Pos
ax = axes[0]
n = n_bins_pos
ax.plot(bin_edges, kappa_pos_fid, c='black')
ax.errorbar(bin_edges, kappa_pos_fid, yerr=np.sqrt(np.diag(kappa_pos_cov)) / np.sqrt(924), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_edges, kappa_pos_derivs_mean[0:n], 
            yerr=np.sqrt(np.diag(kappa_pos_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_edges, kappa_pos_derivs_mean[n:2*n], 
            yerr=np.sqrt(np.diag(kappa_pos_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_edges, kappa_pos_derivs_mean[0:n] - np.sqrt(np.diag(kappa_pos_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), kappa_pos_derivs_mean[0:n] + np.sqrt(np.diag(kappa_pos_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), alpha=0.3)
ax.fill_between(bin_edges, kappa_pos_derivs_mean[n:2*n] - np.sqrt(np.diag(kappa_pos_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), kappa_pos_derivs_mean[n:2*n] + np.sqrt(np.diag(kappa_pos_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$C^{p,n}(\kappa)$')
ax.set_title('Defect CDF+')
ax.set_xticks(np.array([-0.01, 0, 0.01]))
ax.grid(True, alpha=0.3)

# Neg
ax = axes[1]
n = n_bins_neg
ax.plot(bin_edges, kappa_neg_fid, c='black')
ax.errorbar(bin_edges, kappa_neg_fid, yerr=np.sqrt(np.diag(kappa_neg_cov)) / np.sqrt(924), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_edges, kappa_neg_derivs_mean[0:n], 
            yerr=np.sqrt(np.diag(kappa_neg_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_edges, kappa_neg_derivs_mean[n:2*n], 
            yerr=np.sqrt(np.diag(kappa_neg_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_edges, kappa_neg_derivs_mean[0:n] - np.sqrt(np.diag(kappa_neg_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), kappa_neg_derivs_mean[0:n] + np.sqrt(np.diag(kappa_neg_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), alpha=0.3)
ax.fill_between(bin_edges, kappa_neg_derivs_mean[n:2*n] - np.sqrt(np.diag(kappa_neg_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), kappa_neg_derivs_mean[n:2*n] + np.sqrt(np.diag(kappa_neg_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), alpha=0.3)
ax.set_xlabel(r'$\kappa$')
ax.set_title('Defect CDF-')
ax.set_xticks(np.array([-0.01, 0, 0.01]))
ax.grid(True, alpha=0.3)

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 0.55))
plt.subplots_adjust(wspace=0)
plt.savefig('outfiles/derivs_dpdf.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. NNs
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
n = n_bins_nn

# 2NN Same
ax = axes[0]
ax.plot(bin_edges2, nn_pos_same2_fid, c='black')
ax.errorbar(bin_edges2, nn_pos_same2_fid, yerr=np.sqrt(np.diag(nn_pos_same2_cov)) / np.sqrt(924), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_edges2, nn_pos_same2_derivs_mean[0:n], 
            yerr=np.sqrt(np.diag(nn_pos_same2_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_edges2, nn_pos_same2_derivs_mean[n:2*n], 
            yerr=np.sqrt(np.diag(nn_pos_same2_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_edges2, nn_pos_same2_derivs_mean[0:n] - np.sqrt(np.diag(nn_pos_same2_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), nn_pos_same2_derivs_mean[0:n] + np.sqrt(np.diag(nn_pos_same2_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), alpha=0.3)
ax.fill_between(bin_edges2, nn_pos_same2_derivs_mean[n:2*n] - np.sqrt(np.diag(nn_pos_same2_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), nn_pos_same2_derivs_mean[n:2*n] + np.sqrt(np.diag(nn_pos_same2_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), alpha=0.3)
ax.set_title('2NN Same CDF')
ax.set_ylabel(r'$C^{pp,pn}_{\rm 2nd}(\theta)$')
ax.set_xlabel('Distance (arcmin)')
ax.grid(True, alpha=0.3)

# 2NN Opp
ax = axes[1]
ax.plot(bin_edges2, nn_pos_opp2_fid, c='black')
ax.errorbar(bin_edges2, nn_pos_opp2_fid, yerr=np.sqrt(np.diag(nn_pos_opp2_cov)) / np.sqrt(924), fmt='|', label='Fiducial', c='black')
ax.errorbar(bin_edges2, nn_pos_opp2_derivs_mean[0:n], 
            yerr=np.sqrt(np.diag(nn_pos_opp2_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\Omega_m$')
ax.errorbar(bin_edges2, nn_pos_opp2_derivs_mean[n:2*n], 
            yerr=np.sqrt(np.diag(nn_pos_opp2_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), 
            fmt='|', label=r'Derv. w.r.t $\sigma_8$')
ax.fill_between(bin_edges2, nn_pos_opp2_derivs_mean[0:n] - np.sqrt(np.diag(nn_pos_opp2_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), nn_pos_opp2_derivs_mean[0:n] + np.sqrt(np.diag(nn_pos_opp2_derivs_cov[0:n, 0:n])) / np.sqrt(N_samples), alpha=0.3)
ax.fill_between(bin_edges2, nn_pos_opp2_derivs_mean[n:2*n] - np.sqrt(np.diag(nn_pos_opp2_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), nn_pos_opp2_derivs_mean[n:2*n] + np.sqrt(np.diag(nn_pos_opp2_derivs_cov[n:2*n, n:2*n])) / np.sqrt(N_samples), alpha=0.3)
ax.set_title('2NN Opp CDF')
ax.set_xlabel('Distance (arcmin)')
ax.grid(True, alpha=0.3)

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 0.55))
plt.subplots_adjust(wspace=0)
plt.savefig('outfiles/derivs_nns.png', dpi=300, bbox_inches='tight')
plt.close()