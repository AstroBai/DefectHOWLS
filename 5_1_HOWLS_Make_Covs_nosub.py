from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



DATA_DIR = "/cephfs/jbai/HOWLS_Data"

SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

# =======================================================================
N = 1101
kappa_list = []
kappa_pos_list = []
kappa_neg_list = []
nn_pos_same1_list = []
nn_neg_same1_list = []
nn_pos_opp1_list = []
nn_neg_opp1_list = []
nn_pos_same2_list = []
nn_neg_same2_list = []
nn_pos_opp2_list = []
nn_neg_opp2_list = []
nn_pos_same4_list = []
nn_neg_same4_list = []
nn_pos_opp4_list = []
nn_neg_opp4_list = []
nn_pos_same8_list = []
nn_neg_same8_list = []
nn_pos_opp8_list = []
nn_neg_opp8_list = []
kk_list = []
pp_list = []
nn_list = []
pn_list = []    


for i in tqdm(range(N)):
    try:
        data = np.load(f"{SLICS_DIR}/SLICS_stats_s467_th_nosub_{i}.npz")
        bin_edges = data['bin_edges']
        bin_edges1 = data['bin_edges1']
        bin_edges2 = data['bin_edges2']
        bin_edges4 = data['bin_edges4']
        bin_edges8 = data['bin_edges8']
        theta = data['theta']
        kk = data['kk']
        pp = data['pp']
        nn = data['nn']
        pn = data['pn']
        kappa = data['kappa']
        kappa_pos_cum = data['kappa_pos_cum']
        kappa_neg_cum = data['kappa_neg_cum']
        nn_pos_same1_cum = data['nn_pos_same1_cum']
        nn_neg_same1_cum = data['nn_neg_same1_cum']
        nn_pos_opp1_cum = data['nn_pos_opp1_cum']
        nn_neg_opp1_cum = data['nn_neg_opp1_cum']
        nn_pos_same2_cum = data['nn_pos_same2_cum']
        nn_neg_same2_cum = data['nn_neg_same2_cum']
        nn_pos_opp2_cum = data['nn_pos_opp2_cum']
        nn_neg_opp2_cum = data['nn_neg_opp2_cum']
        nn_pos_same4_cum = data['nn_pos_same4_cum']
        nn_neg_same4_cum = data['nn_neg_same4_cum']
        nn_pos_opp4_cum = data['nn_pos_opp4_cum']
        nn_neg_opp4_cum = data['nn_neg_opp4_cum']
        nn_pos_same8_cum = data['nn_pos_same8_cum']
        nn_neg_same8_cum = data['nn_neg_same8_cum']
        nn_pos_opp8_cum = data['nn_pos_opp8_cum']
        nn_neg_opp8_cum = data['nn_neg_opp8_cum']
        print(pp)
        pp_list.append(pp)
        nn_list.append(nn)
        pn_list.append(pn)
        kk_list.append(kk)
        kappa_list.append(kappa)
        kappa_pos_list.append(kappa_pos_cum)
        kappa_neg_list.append(kappa_neg_cum)
        nn_pos_same1_list.append(nn_pos_same1_cum)
        nn_neg_same1_list.append(nn_neg_same1_cum)
        nn_pos_opp1_list.append(nn_pos_opp1_cum)
        nn_neg_opp1_list.append(nn_neg_opp1_cum)
        nn_pos_same2_list.append(nn_pos_same2_cum)
        nn_neg_same2_list.append(nn_neg_same2_cum)
        nn_pos_opp2_list.append(nn_pos_opp2_cum)
        nn_neg_opp2_list.append(nn_neg_opp2_cum)
        nn_pos_same4_list.append(nn_pos_same4_cum)
        nn_neg_same4_list.append(nn_neg_same4_cum)
        nn_pos_opp4_list.append(nn_pos_opp4_cum)
        nn_neg_opp4_list.append(nn_neg_opp4_cum)
        nn_pos_same8_list.append(nn_pos_same8_cum)
        nn_neg_same8_list.append(nn_neg_same8_cum)
        nn_pos_opp8_list.append(nn_pos_opp8_cum)
        nn_neg_opp8_list.append(nn_neg_opp8_cum)    
    except FileNotFoundError:
        print(f"File for realisation {i} not found, skipping.")

pp_list = np.array(pp_list)
nn_list = np.array(nn_list)
pn_list = np.array(pn_list)
kk_list = np.array(kk_list)
kappa_list =  np.array(kappa_list)
kappa_pos_list =  np.array(kappa_pos_list)
kappa_neg_list =  np.array(kappa_neg_list)
nn_pos_same1_list = np.array(nn_pos_same1_list)
nn_neg_same1_list = np.array(nn_neg_same1_list)
nn_pos_opp1_list = np.array(nn_pos_opp1_list)
nn_neg_opp1_list = np.array(nn_neg_opp1_list)
nn_pos_same2_list = np.array(nn_pos_same2_list)
nn_neg_same2_list = np.array(nn_neg_same2_list)
nn_pos_opp2_list = np.array(nn_pos_opp2_list)
nn_neg_opp2_list = np.array(nn_neg_opp2_list)
nn_pos_same4_list = np.array(nn_pos_same4_list)
nn_neg_same4_list = np.array(nn_neg_same4_list)
nn_pos_opp4_list = np.array(nn_pos_opp4_list)
nn_neg_opp4_list = np.array(nn_neg_opp4_list)
nn_pos_same8_list = np.array(nn_pos_same8_list)
nn_neg_same8_list = np.array(nn_neg_same8_list)
nn_pos_opp8_list = np.array(nn_pos_opp8_list)
nn_neg_opp8_list = np.array(nn_neg_opp8_list)

pp_mean = np.mean(pp_list, axis=0)
nn_mean = np.mean(nn_list, axis=0)
pn_mean = np.mean(pn_list, axis=0)
dcorr_mean = np.concatenate([pp_mean, nn_mean, pn_mean])
kk_mean = np.mean(kk_list, axis=0)
kappa_mean = np.mean(kappa_list, axis=0)
kappa_pos_mean = np.mean(kappa_pos_list, axis=0)
kappa_neg_mean = np.mean(kappa_neg_list, axis=0)
nn_pos_same1_mean = np.mean(nn_pos_same1_list, axis=0)
nn_neg_same1_mean = np.mean(nn_neg_same1_list, axis=0)
nn_pos_opp1_mean = np.mean(nn_pos_opp1_list, axis=0)
nn_neg_opp1_mean = np.mean(nn_neg_opp1_list, axis=0)
nn_pos_same2_mean = np.mean(nn_pos_same2_list, axis=0)
nn_neg_same2_mean = np.mean(nn_neg_same2_list, axis=0)
nn_pos_opp2_mean = np.mean(nn_pos_opp2_list, axis=0)
nn_neg_opp2_mean = np.mean(nn_neg_opp2_list, axis=0)
nn_pos_same4_mean = np.mean(nn_pos_same4_list, axis=0)
nn_neg_same4_mean = np.mean(nn_neg_same4_list, axis=0)
nn_pos_opp4_mean = np.mean(nn_pos_opp4_list, axis=0)
nn_neg_opp4_mean = np.mean(nn_neg_opp4_list, axis=0)
nn_pos_same8_mean = np.mean(nn_pos_same8_list, axis=0)
nn_neg_same8_mean = np.mean(nn_neg_same8_list, axis=0)
nn_pos_opp8_mean = np.mean(nn_pos_opp8_list, axis=0)
nn_neg_opp8_mean = np.mean(nn_neg_opp8_list, axis=0)
kappa_all_mean = np.concatenate([kappa_pos_mean, kappa_neg_mean])
nn_1_mean = np.concatenate([nn_pos_same1_mean, nn_neg_same1_mean, nn_pos_opp1_mean, nn_neg_opp1_mean])
nn_2_mean = np.concatenate([nn_pos_same2_mean, nn_neg_same2_mean, nn_pos_opp2_mean, nn_neg_opp2_mean])
nn_4_mean = np.concatenate([nn_pos_same4_mean, nn_neg_same4_mean, nn_pos_opp4_mean, nn_neg_opp4_mean])
nn_8_mean = np.concatenate([nn_pos_same8_mean, nn_neg_same8_mean, nn_pos_opp8_mean, nn_neg_opp8_mean])
nn_all_mean = np.concatenate([nn_1_mean, nn_2_mean, nn_4_mean, nn_8_mean])
all_mean = np.concatenate([kappa_pos_mean, kappa_neg_mean, nn_pos_same1_mean, nn_neg_same1_mean, nn_pos_opp1_mean, nn_neg_opp1_mean,
                           nn_pos_same2_mean, nn_neg_same2_mean, nn_pos_opp2_mean, nn_neg_opp2_mean,
                           nn_pos_same4_mean, nn_neg_same4_mean, nn_pos_opp4_mean, nn_neg_opp4_mean,
                           nn_pos_same8_mean, nn_neg_same8_mean, nn_pos_opp8_mean, nn_neg_opp8_mean])
all_and_pdf_mean = np.concatenate([all_mean, kappa_mean])
all_howls_and_kk_mean = np.concatenate([all_and_pdf_mean, kk_mean])
all_and_kk_mean = np.concatenate([all_mean, kk_mean])

pp_cov = np.cov(pp_list, rowvar=False)
nn_cov = np.cov(nn_list, rowvar=False)
pn_cov = np.cov(pn_list, rowvar=False)
dcorr_cov = np.cov(np.hstack([pp_list, nn_list, pn_list]), rowvar=False)
kk_cov = np.cov(kk_list, rowvar=False)
kappa_cov = np.cov(kappa_list, rowvar=False)
kappa_pos_cov = np.cov(kappa_pos_list, rowvar=False)
kappa_neg_cov = np.cov(kappa_neg_list, rowvar=False)
nn_pos_same1_cov = np.cov(nn_pos_same1_list, rowvar=False)
nn_neg_same1_cov = np.cov(nn_neg_same1_list, rowvar=False)
nn_pos_opp1_cov = np.cov(nn_pos_opp1_list, rowvar=False)
nn_neg_opp1_cov = np.cov(nn_neg_opp1_list, rowvar=False)
nn_pos_same2_cov = np.cov(nn_pos_same2_list, rowvar=False)
nn_neg_same2_cov = np.cov(nn_neg_same2_list, rowvar=False)
nn_pos_opp2_cov = np.cov(nn_pos_opp2_list, rowvar=False)
nn_neg_opp2_cov = np.cov(nn_neg_opp2_list, rowvar=False)
nn_pos_same4_cov = np.cov(nn_pos_same4_list, rowvar=False)
nn_neg_same4_cov = np.cov(nn_neg_same4_list, rowvar=False)
nn_pos_opp4_cov = np.cov(nn_pos_opp4_list, rowvar=False)
nn_neg_opp4_cov = np.cov(nn_neg_opp4_list, rowvar=False)
nn_pos_same8_cov = np.cov(nn_pos_same8_list, rowvar=False)
nn_neg_same8_cov = np.cov(nn_neg_same8_list, rowvar=False)
nn_pos_opp8_cov = np.cov(nn_pos_opp8_list, rowvar=False)
nn_neg_opp8_cov = np.cov(nn_neg_opp8_list, rowvar=False)
kappa_all_cov = np.cov(np.hstack([kappa_pos_list, kappa_neg_list]), rowvar=False)
nn_1_cov = np.cov(np.hstack([nn_pos_same1_list, nn_neg_same1_list, nn_pos_opp1_list, nn_neg_opp1_list]), rowvar=False)
nn_2_cov = np.cov(np.hstack([nn_pos_same2_list, nn_neg_same2_list, nn_pos_opp2_list, nn_neg_opp2_list]), rowvar=False)
nn_4_cov = np.cov(np.hstack([nn_pos_same4_list, nn_neg_same4_list, nn_pos_opp4_list, nn_neg_opp4_list]), rowvar=False)
nn_8_cov = np.cov(np.hstack([nn_pos_same8_list, nn_neg_same8_list, nn_pos_opp8_list, nn_neg_opp8_list]), rowvar=False)
nn_all_cov = np.cov(np.hstack([nn_pos_same1_list, nn_neg_same1_list, nn_pos_opp1_list, nn_neg_opp1_list,
                               nn_pos_same2_list, nn_neg_same2_list, nn_pos_opp2_list, nn_neg_opp2_list,
                               nn_pos_same4_list, nn_neg_same4_list, nn_pos_opp4_list, nn_neg_opp4_list,
                               nn_pos_same8_list, nn_neg_same8_list, nn_pos_opp8_list, nn_neg_opp8_list]), rowvar=False)
all_cov = np.cov(np.hstack([kappa_pos_list, kappa_neg_list,
                            nn_pos_same1_list, nn_neg_same1_list, nn_pos_opp1_list, nn_neg_opp1_list,
                            nn_pos_same2_list, nn_neg_same2_list, nn_pos_opp2_list, nn_neg_opp2_list,
                            nn_pos_same4_list, nn_neg_same4_list, nn_pos_opp4_list, nn_neg_opp4_list,
                            nn_pos_same8_list, nn_neg_same8_list, nn_pos_opp8_list, nn_neg_opp8_list]), rowvar=False)
all_and_pdf_cov = np.cov(np.hstack([kappa_pos_list, kappa_neg_list,
                            nn_pos_same1_list, nn_neg_same1_list, nn_pos_opp1_list, nn_neg_opp1_list,
                            nn_pos_same2_list, nn_neg_same2_list, nn_pos_opp2_list, nn_neg_opp2_list,
                            nn_pos_same4_list, nn_neg_same4_list, nn_pos_opp4_list, nn_neg_opp4_list,
                            nn_pos_same8_list, nn_neg_same8_list, nn_pos_opp8_list, nn_neg_opp8_list,
                            kappa_list]), rowvar=False)
all_howls_and_kk_cov = np.cov(np.hstack([kappa_pos_list, kappa_neg_list,
                            nn_pos_same1_list, nn_neg_same1_list, nn_pos_opp1_list, nn_neg_opp1_list,
                            nn_pos_same2_list, nn_neg_same2_list, nn_pos_opp2_list, nn_neg_opp2_list,
                            nn_pos_same4_list, nn_neg_same4_list, nn_pos_opp4_list, nn_neg_opp4_list,
                            nn_pos_same8_list, nn_neg_same8_list, nn_pos_opp8_list, nn_neg_opp8_list,
                            kappa_list, kk_list]), rowvar=False)
all_and_kk_cov = np.cov(np.hstack([kappa_pos_list, kappa_neg_list,
                            nn_pos_same1_list, nn_neg_same1_list, nn_pos_opp1_list, nn_neg_opp1_list,
                            nn_pos_same2_list, nn_neg_same2_list, nn_pos_opp2_list, nn_neg_opp2_list,
                            nn_pos_same4_list, nn_neg_same4_list, nn_pos_opp4_list, nn_neg_opp4_list,
                            nn_pos_same8_list, nn_neg_same8_list, nn_pos_opp8_list, nn_neg_opp8_list,
                            kk_list]), rowvar=False)



np.savez(f"{SLICS_DIR}/SLICS_stats_cov_mean_s467_th_nosub.npz",
         bin_edges=bin_edges,
         bin_edges1=bin_edges1,
         bin_edges2=bin_edges2,
         bin_edges4=bin_edges4,
         bin_edges8=bin_edges8,
            theta=theta,
            kk_mean=kk_mean,
            pp_mean=pp_mean,
            nn_mean=nn_mean,
            pn_mean=pn_mean,
            dcorr_mean=dcorr_mean,
            kappa_mean=kappa_mean,
         kappa_pos_mean=kappa_pos_mean,
         kappa_neg_mean=kappa_neg_mean,
         nn_pos_same1_mean=nn_pos_same1_mean,
         nn_neg_same1_mean=nn_neg_same1_mean,
         nn_pos_opp1_mean=nn_pos_opp1_mean,
         nn_neg_opp1_mean=nn_neg_opp1_mean,
         nn_pos_same2_mean=nn_pos_same2_mean,
         nn_neg_same2_mean=nn_neg_same2_mean,
         nn_pos_opp2_mean=nn_pos_opp2_mean,
         nn_neg_opp2_mean=nn_neg_opp2_mean,
         nn_pos_same4_mean=nn_pos_same4_mean,
         nn_neg_same4_mean=nn_neg_same4_mean,
         nn_pos_opp4_mean=nn_pos_opp4_mean,
         nn_neg_opp4_mean=nn_neg_opp4_mean,
         nn_pos_same8_mean=nn_pos_same8_mean,
         nn_neg_same8_mean=nn_neg_same8_mean,
         nn_pos_opp8_mean=nn_pos_opp8_mean,
         nn_neg_opp8_mean=nn_neg_opp8_mean,
         kappa_all_mean=kappa_all_mean,
         nn_1_mean=nn_1_mean,
         nn_2_mean=nn_2_mean,
         nn_4_mean=nn_4_mean,
         nn_8_mean=nn_8_mean,
         nn_all_mean=nn_all_mean,
            all_mean=all_mean,
            all_and_pdf_mean=all_and_pdf_mean,
            all_howls_and_kk_mean=all_howls_and_kk_mean,
            all_and_kk_mean=all_and_kk_mean,
        kappa_cov=kappa_cov,
         kappa_pos_cov=kappa_pos_cov,
         kappa_neg_cov=kappa_neg_cov,
         nn_pos_same1_cov=nn_pos_same1_cov,
         nn_neg_same1_cov=nn_neg_same1_cov,
         nn_pos_opp1_cov=nn_pos_opp1_cov,
         nn_neg_opp1_cov=nn_neg_opp1_cov,
         nn_pos_same2_cov=nn_pos_same2_cov,
         nn_neg_same2_cov=nn_neg_same2_cov,
         nn_pos_opp2_cov=nn_pos_opp2_cov,
         nn_neg_opp2_cov=nn_neg_opp2_cov,
         nn_pos_same4_cov=nn_pos_same4_cov,
         nn_neg_same4_cov=nn_neg_same4_cov,
         nn_pos_opp4_cov=nn_pos_opp4_cov,
         nn_neg_opp4_cov=nn_neg_opp4_cov,
         nn_pos_same8_cov=nn_pos_same8_cov,
         nn_neg_same8_cov=nn_neg_same8_cov,
         nn_pos_opp8_cov=nn_pos_opp8_cov,
         nn_neg_opp8_cov=nn_neg_opp8_cov,
         kappa_all_cov=kappa_all_cov,
         nn_1_cov=nn_1_cov,
         nn_2_cov=nn_2_cov,
         nn_4_cov=nn_4_cov,
         nn_8_cov=nn_8_cov,
         nn_all_cov=nn_all_cov,
         all_cov=all_cov,
         kk_cov=kk_cov,
         all_and_pdf_cov=all_and_pdf_cov,
         all_howls_and_kk_cov=all_howls_and_kk_cov,
            all_and_kk_cov=all_and_kk_cov,
            pp_cov=pp_cov,
            nn_cov=nn_cov,
            pn_cov=pn_cov,
            dcorr_cov=dcorr_cov
         )

def filter_data_vector(mean_vec, cov_mat):
    """
    Applies S/N and correlation cuts to identify indices of bins to keep.
    Returns: mask array (boolean)
    """
    sigma_thresh = 2.0
    sigma = np.sqrt(np.diag(cov_mat))
    
    # S/N cut
    mask = np.abs(mean_vec) >= sigma_thresh * sigma
    
    if mask.sum() == 0:
        return mask # Empty mask
        
    cov_trim = cov_mat[np.ix_(mask, mask)]
    
    # Correlation cut
    sigma_trim = np.sqrt(np.diag(cov_trim))
    corr = cov_trim / np.outer(sigma_trim, sigma_trim)
    
    corr_mask = np.ones(corr.shape[0], dtype=bool)
    indices_in_trim = np.arange(corr.shape[0])
    
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if np.abs(corr[i, j]) >= 0.9999:
                # Use mean/sigma as S/N proxy for deciding which to keep
                sn_i = np.abs(mean_vec[mask][i]) / sigma[mask][i]
                sn_j = np.abs(mean_vec[mask][j]) / sigma[mask][j]
                if sn_i < sn_j:
                    corr_mask[i] = False
                else:
                    corr_mask[j] = False
                    
    # Final mask mapping back to original indices
    final_indices = np.where(mask)[0][corr_mask]
    final_mask = np.zeros_like(mask)
    if len(final_indices) > 0:
        final_mask[final_indices] = True
        
    return final_mask

# Plot Ridgeline Distributions for SLICS with KL divergence
quantities = {
    'KK': (kk_list, kk_mean, kk_cov),
    'PP': (pp_list, pp_mean, pp_cov),
    'NN': (nn_list, nn_mean, nn_cov),
    'PN': (pn_list, pn_mean, pn_cov),
    'Kappa': (kappa_list, kappa_mean, kappa_cov),
    'Kappa_Pos': (kappa_pos_list, kappa_pos_mean, kappa_pos_cov),
    'Kappa_Neg': (kappa_neg_list, kappa_neg_mean, kappa_neg_cov),
    'NN_Pos_Same_1': (nn_pos_same1_list, nn_pos_same1_mean, nn_pos_same1_cov),
    'NN_Neg_Same_1': (nn_neg_same1_list, nn_neg_same1_mean, nn_neg_same1_cov),
    'NN_Pos_Opp_1': (nn_pos_opp1_list, nn_pos_opp1_mean, nn_pos_opp1_cov),
    'NN_Neg_Opp_1': (nn_neg_opp1_list, nn_neg_opp1_mean, nn_neg_opp1_cov),
    'NN_Pos_Same_2': (nn_pos_same2_list, nn_pos_same2_mean, nn_pos_same2_cov),
    'NN_Neg_Same_2': (nn_neg_same2_list, nn_neg_same2_mean, nn_neg_same2_cov),
    'NN_Pos_Opp_2': (nn_pos_opp2_list, nn_pos_opp2_mean, nn_pos_opp2_cov),
    'NN_Neg_Opp_2': (nn_neg_opp2_list, nn_neg_opp2_mean, nn_neg_opp2_cov),
    'NN_Pos_Same_4': (nn_pos_same4_list, nn_pos_same4_mean, nn_pos_same4_cov),
    'NN_Neg_Same_4': (nn_neg_same4_list, nn_neg_same4_mean, nn_neg_same4_cov),
    'NN_Pos_Opp_4': (nn_pos_opp4_list, nn_pos_opp4_mean, nn_pos_opp4_cov),
    'NN_Neg_Opp_4': (nn_neg_opp4_list, nn_neg_opp4_mean, nn_neg_opp4_cov),
    'NN_Pos_Same_8': (nn_pos_same8_list, nn_pos_same8_mean, nn_pos_same8_cov),
    'NN_Neg_Same_8': (nn_neg_same8_list, nn_neg_same8_mean, nn_neg_same8_cov),
    'NN_Pos_Opp_8': (nn_pos_opp8_list, nn_pos_opp8_mean, nn_pos_opp8_cov),
    'NN_Neg_Opp_8': (nn_neg_opp8_list, nn_neg_opp8_mean, nn_neg_opp8_cov)
}

print("Plotting SLICS ridgeline distributions with KL divergence...")
from scipy.stats import norm, entropy

def calculate_kl(data_vec):
    """Calculates KL(P || Q) where P is data, Q is Gaussian approximation."""
    mu, std = np.mean(data_vec), np.std(data_vec)
    if std == 0: return 0.0
    
    # Discretize for KL calc
    hist, bin_edges = np.histogram(data_vec, bins=50, density=True)
    pk = hist * np.diff(bin_edges) # probabilities
    
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    qk = norm.pdf(x, mu, std) * np.diff(bin_edges)
    
    # Avoid log(0)
    pk = np.where(pk == 0, 1e-10, pk)
    qk = np.where(qk == 0, 1e-10, qk)
    
    return np.sum(pk * np.log(pk / qk))

# Redefine plot_ridgeline to support KL annotations directly
def plot_ridgeline_kl(data_matrix, mask, title, filename_suffix):
    n_samples, n_bins = data_matrix.shape
    kept_indices = np.where(mask)[0]
    
    if len(kept_indices) == 0:
        return

    bin_indices = kept_indices
    selected_data = data_matrix[:, bin_indices]
    n_plot_bins = len(bin_indices)
    
    fig, ax = plt.subplots(figsize=(max(6, n_plot_bins * 0.4), 8))
    
    overlap = 0.5
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot_bins))

    for i, bin_idx in enumerate(bin_indices):
        x_data = selected_data[:, i]
        
        # Calculate KL
        kl = calculate_kl(x_data)

        if np.var(x_data) == 0:
            continue
            
        kde = gaussian_kde(x_data)
        y_grid = np.linspace(x_data.min(), x_data.max(), 100)
        x_density = kde(y_grid)
        
        if x_density.max() > 0:
            x_density = x_density / x_density.max() * (1 + overlap)
        
        baseline = i
        ax.fill_betweenx(y_grid, baseline, baseline + x_density, color=colors[i], alpha=0.7, zorder=n_plot_bins-i)
        ax.plot(baseline + x_density, y_grid, color='white', lw=1, zorder=n_plot_bins-i)
        
        # Annotate KL at the bottom of the graph (near X axis baseline value)
        ax.text(baseline + 0.1, y_grid.min(), f"KL={kl:.3f}", rotation=90, va='top', fontsize=7, zorder=n_plot_bins-i+1)

    ax.set_xticks(np.arange(len(bin_indices)))
    ax.set_xticklabels([f"Bin {b}" for b in bin_indices], rotation=90)
    ax.set_title(title)
    ax.set_xlabel("Bin Index")
    ax.set_ylabel("Value")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    plt.tight_layout()
    os.makedirs("outfiles/Ridgelines_SLICS_KL", exist_ok=True)
    plt.savefig(f"outfiles/Ridgelines_SLICS_KL/{filename_suffix}.png", dpi=300)
    plt.close()

for name, (data_mat, mean_vec, cov_mat) in quantities.items():
    mask = filter_data_vector(mean_vec, cov_mat)
    try:
        plot_ridgeline_kl(data_mat, mask, f"SLICS Distribution of {name} (Filtered)", f"{name}_ridgeline_SLICS_KL")
    except Exception as e:
        print(f"Failed to plot ridgeline KL for {name}: {e}")