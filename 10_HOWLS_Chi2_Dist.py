import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from tqdm import tqdm
from scipy.stats import norm, chi2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.formatter.use_mathtext'] = True

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"
OUT_DIR = "outfiles/Chi2Tests"
os.makedirs(OUT_DIR, exist_ok=True)

def calculate_kl(data_vec):
    """Calculates KL(P || Q) where P is data, Q is Gaussian approximation."""
    mu, std = np.mean(data_vec), np.std(data_vec)
    if std == 0: return 0.0
    
    # Discretize for KL calc
    hist, bin_edges = np.histogram(data_vec, bins=50, density=True)
    pk = hist * np.diff(bin_edges) # probabilities
    
    # Ensure bin centers
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    qk = norm.pdf(x, mu, std) * np.diff(bin_edges)
    
    # Avoid log(0) and division by zero
    pk = np.where(pk == 0, 1e-10, pk)
    qk = np.where(qk == 0, 1e-10, qk)
    
    return np.sum(pk * np.log(pk / qk))

def load_slics_samples(slics_dir, n_files=1101):
    print("Loading SLICS samples...")
    
    # Initialize lists
    keys = ['kappa', 'kappa_pos_cum', 'kappa_neg_cum', 
            'nn_pos_same1_cum', 'nn_neg_same1_cum', 'nn_pos_opp1_cum', 'nn_neg_opp1_cum',
            'nn_pos_same2_cum', 'nn_neg_same2_cum', 'nn_pos_opp2_cum', 'nn_neg_opp2_cum',
            'nn_pos_same4_cum', 'nn_neg_same4_cum', 'nn_pos_opp4_cum', 'nn_neg_opp4_cum',
            'nn_pos_same8_cum', 'nn_neg_same8_cum', 'nn_pos_opp8_cum', 'nn_neg_opp8_cum',
            'kk', 'pp', 'nn', 'pn']
            
    samples = {k: [] for k in keys}
    
    count = 0
    # Loading samples
    for i in tqdm(range(n_files), desc="Loading SLICS"):
        for subfield in range(4):
            try:
                filepath = f"{slics_dir}/SLICS_stats_s467_th_{i}_subfield{subfield}.npz"
                if not os.path.exists(filepath):
                    continue
                
                with np.load(filepath) as data:
                    # Special handling for kappa truncation if needed, 
                    # but usually it's just an array in the file
                    if 'kappa' in data:
                        samples['kappa'].append(data['kappa'])
                    
                    # Direct mapping for others
                    for k in keys:
                        if k == 'kappa': continue
                        if k in data:
                            samples[k].append(data[k])
                count += 1
            except Exception as e:
                pass
    
    print(f"Loaded {count} samples.")

    final_samples = {}
    # Name mapping to match code variables
    name_map = {
        'kappa': 'kappa', 'kk': 'kk', 'pp': 'pp', 'nn': 'nn', 'pn': 'pn',
        'kappa_pos_cum': 'kappa_pos', 'kappa_neg_cum': 'kappa_neg',
        'nn_pos_same1_cum': 'nn_pos_same1', 'nn_neg_same1_cum': 'nn_neg_same1',
        'nn_pos_opp1_cum': 'nn_pos_opp1', 'nn_neg_opp1_cum': 'nn_neg_opp1',
        'nn_pos_same2_cum': 'nn_pos_same2', 'nn_neg_same2_cum': 'nn_neg_same2',
        'nn_pos_opp2_cum': 'nn_pos_opp2', 'nn_neg_opp2_cum': 'nn_neg_opp2',
        'nn_pos_same4_cum': 'nn_pos_same4', 'nn_neg_same4_cum': 'nn_neg_same4',
        'nn_pos_opp4_cum': 'nn_pos_opp4', 'nn_neg_opp4_cum': 'nn_neg_opp4',
        'nn_pos_same8_cum': 'nn_pos_same8', 'nn_neg_same8_cum': 'nn_neg_same8',
        'nn_pos_opp8_cum': 'nn_pos_opp8', 'nn_neg_opp8_cum': 'nn_neg_opp8'
    }
    
    for k, v in samples.items():
        if len(v) > 0:
            final_samples[name_map[k]] = np.array(v)
            
    return final_samples

def get_kept_mask(name, fid_vec, cov, samples=None, sigma_thresh=2.0):
    # 1. S/N Filter (> 2.0)
    sigma = np.sqrt(np.diag(cov))
    sigma[sigma == 0] = 1.0 # Avoid division by zero
    
    # Calculate S/N
    # Safe division
    sn = np.abs(fid_vec) / sigma
    mask = sn > sigma_thresh
    
    if mask.sum() == 0:
        return mask

    # 1.5 KL Filter
    if samples is not None:
        print(f"Applying KL filter for {name}...")
        kl_values = np.zeros(fid_vec.shape[0])
        # Calculate KL only for S/N passed bins
        for i in range(fid_vec.shape[0]):
             if mask[i]:
                 # samples shape check: (n_realizations, n_bins)
                 # We need to ensure we are accessing the right column
                 if i < samples.shape[1]:
                    kl_values[i] = calculate_kl(samples[:, i])
        
        mask = mask & (kl_values <= 0.2) # Threshold 0.1 consistent with Fisher script
        
        if mask.sum() == 0:
            print(f"WARNING: {name}: All bins removed by KL filter.")
            return mask

    # 2. Correlation Filter (remove > 0.9999)
    # We only check correlation among the bins surviving the S/N cut
    cov_trim = cov[np.ix_(mask, mask)]
    sigma_trim = np.sqrt(np.diag(cov_trim))
    sigma_trim[sigma_trim == 0] = 1.0
    
    corr = cov_trim / np.outer(sigma_trim, sigma_trim)
    
    corr_mask = np.ones(corr.shape[0], dtype=bool)
    # Check upper triangle
    # We iterate over the REDUCED matrix indices
    for i in range(corr.shape[0]):
        if not corr_mask[i]: continue
        for j in range(i + 1, corr.shape[0]):
            if not corr_mask[j]: continue
            if np.abs(corr[i, j]) >= 0.9999:
                # Remove the bin with lower S/N
                # Need to map back to original indices to check S/N
                indices_in_full = np.where(mask)[0]
                idx_i = indices_in_full[i]
                idx_j = indices_in_full[j]
                
                s_i = sigma[idx_i] if sigma[idx_i] > 0 else 1.0
                s_j = sigma[idx_j] if sigma[idx_j] > 0 else 1.0
                
                sn_i = np.abs(fid_vec[idx_i]) / s_i
                sn_j = np.abs(fid_vec[idx_j]) / s_j
                
                if sn_i <= sn_j:
                    corr_mask[i] = False
                else:
                    corr_mask[j] = False

    # Apply correlation mask to the S/N mask
    # corr_mask is boolean for the ALREADY MASKED subset
    if corr_mask.sum() < len(corr_mask):
         # Get indices of S/N passed bins
        sn_indices = np.where(mask)[0]
        # Keep only those that passed correlation
        final_indices = sn_indices[corr_mask]
        
        new_mask = np.zeros_like(mask)
        new_mask[final_indices] = True
        return new_mask

    return mask

def main():
    # 1. Load Samples
    # Note: Ensure you have enough memory. If it crashes, increase allocation or reduce load.
    samples_dict = load_slics_samples(SLICS_DIR)
    
    # 2. Define Groups
    # panel name -> list of keys
    groups = {
        'dcdf': ['kappa_pos', 'kappa_neg'],
        'knn': ['nn_pos_same1', 'nn_neg_same1', 'nn_pos_opp1', 'nn_neg_opp1',
                'nn_pos_same2', 'nn_neg_same2', 'nn_pos_opp2', 'nn_neg_opp2',
                'nn_pos_same4', 'nn_neg_same4', 'nn_pos_opp4', 'nn_neg_opp4',
                'nn_pos_same8', 'nn_neg_same8', 'nn_pos_opp8', 'nn_neg_opp8'],
        # Order for kNN often specific, but here we just concatenate.
        'pdf': ['kappa'],
        '2pcf': ['kk']
    }
    
    # Display Names map
    display_names = {
        'dcdf': 'D-CDF',
        'knn': 'kNN',
        'pdf': 'PDF',
        '2pcf': r'$\kappa$-2PCF'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    axes = axes.flatten()
    
    # Order: D-CDF, kNN, PDF, 2PCF
    # Map keys to index
    ordered_keys = ['dcdf', 'knn', 'pdf', '2pcf']
    
    for idx, key in enumerate(ordered_keys):
        ax = axes[idx]
        group_keys = groups[key]
        display_name = display_names[key]
        
        print(f"\nProcessing {display_name}...")
        
        # Check if all keys exist
        missing = [k for k in group_keys if k not in samples_dict]
        if missing:
            print(f"  Missing keys: {missing}. Skipping.")
            continue
            
        # Combine Samples
        # List of (N_samples, N_bins_sub)
        group_samples_list = [samples_dict[k] for k in group_keys]
        
        # Concatenate along axis 1 (bins)
        D_matrix = np.hstack(group_samples_list) # (N_samples, N_total_bins)
        
        # Calculate Mean and Cov from Samples
        mu = np.mean(D_matrix, axis=0) # (N_total_bins,)
        C_sample = np.cov(D_matrix, rowvar=False) # (N_total_bins, N_total_bins) (Patch Covariance)
        
        # Area Scaling for Filtering
        # in Fisher forecast, we filter based on Survey S/N
        # C_survey = C_sample * (Area_patch / Area_survey)
        area_scaling = 25 / 15000 
        C_survey = C_sample * area_scaling
        
        print(f"  Total bins: {len(mu)}")
        print(f"  Total samples: {D_matrix.shape[0]}")
        
        # Calculate Mask using Survey Covariance for S/N check
        mask = get_kept_mask(key, mu, C_survey, samples=D_matrix)
        n_kept = np.sum(mask)
        print(f"  Kept {n_kept} bins after criteria.")
        
        if n_kept == 0:
            print("  No bins kept!")
            ax.text(0.5, 0.5, 'No bins kept', ha='center', va='center')
            continue
            
        # Apply Mask
        D_kept = D_matrix[:, mask]
        mu_kept = mu[mask]
        
        # Chi2 Calculation should use the SAMPLE Covariance (Patch level) because D_matrix consists of patch samples
        C_kept = C_sample[np.ix_(mask, mask)]
        
        # Invert Covariance
        try:
            # Using pinv is safer for numerical stability
            inv_C = np.linalg.inv(C_kept)
        except np.linalg.LinAlgError:
            print("  Covariance singular, using pseudo-inverse")
            inv_C = np.linalg.pinv(C_kept)
        
        # Compute Mahalanobis Distance Squared (Chi2) for each sample
        diff = D_kept - mu_kept # (N_samples, N_kept)
        
        # y_i = (diff_i) C^-1 (diff_i)^T
        # Vectorized: sum( (diff @ inv_C) * diff, axis=1 )
        y = np.sum((diff @ inv_C) * diff, axis=1)
        
        # Plot Histogram
        # Normalize to density
        n_bins_smape = 40
        sns.histplot(y, ax=ax, stat='density', element="step", fill=True, alpha=0.3, label='SLICS Samples', bins=n_bins_smape)
        
        # Plot Theoretical Chi2
        # Range of x
        x_min, x_max = np.min(y), np.max(y)
        # Extend range slightly for theoretical plot
        x_range = np.linspace(chi2.ppf(0.00001, n_kept), chi2.ppf(0.99999, n_kept), 200)
        
        ax.plot(x_range, chi2.pdf(x_range, n_kept), 'r-', lw=2, label=rf'Theory')
        
        # Calculate SMAPE between empirical histogram and theoretical PDF
        # We use the same bins as the histogram to estimate density
        # Use a reasonable number of bins for the SMAPE calculation
        
        hist_vals, bin_edges = np.histogram(y, bins=n_bins_smape, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Theoretical PDF at bin centers
        P_chi2 = chi2.pdf(bin_centers, n_kept)
        P_obs = hist_vals
        
        # Avoid division by zero in S_i
        denominator = (np.abs(P_obs) + np.abs(P_chi2))
        
        # S_i calculation
        # Handle 0/0 case if any
        with np.errstate(divide='ignore', invalid='ignore'):
            S_i = np.abs(P_obs - P_chi2) / denominator
        S_i = np.nan_to_num(S_i, nan=0.0)
        
        # Weight w_i = P_chi2(y_i)
        w_i = P_chi2
        
        # Weighted Average
        if np.sum(w_i) > 0:
            smape = np.sum(w_i * S_i) / np.sum(w_i)
        else:
            smape = 0.0
        
        # Add SMAPE to plot
        ax.text(0.95, 0.85, f'{display_name}, SMAPE = {smape:.3f}', 
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        
        # ax.set_title(display_name, fontsize=14)
        ax.set_xlabel(r'$\chi^2$')
        # only keep 4 yticks, with .3f formatting
        yticks = np.linspace(0, ax.get_ylim()[1], 4)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{y:.3f}' for y in yticks])
        # ax.set_ylabel('')
        # ax.set_yticks([]) # Hide y-axis ticks for cleaner look
        # ax.legend()
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    output_path = f"{OUT_DIR}/Chi2_Distribution_Comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved plot to {output_path}")

if __name__ == "__main__":
    main()
