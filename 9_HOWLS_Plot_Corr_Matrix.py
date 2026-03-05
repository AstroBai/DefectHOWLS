import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from scipy.stats import norm, gaussian_kde
from matplotlib.colors import SymLogNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"
OUT_DIR = "outfiles/CorrMatrices"
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
    print("Loading SLICS samples for KL calculation...")
    
    # Initialize lists
    keys = ['kappa', 'kappa_pos_cum', 'kappa_neg_cum', 
            'nn_pos_same1_cum', 'nn_neg_same1_cum', 'nn_pos_opp1_cum', 'nn_neg_opp1_cum',
            'nn_pos_same2_cum', 'nn_neg_same2_cum', 'nn_pos_opp2_cum', 'nn_neg_opp2_cum',
            'nn_pos_same4_cum', 'nn_neg_same4_cum', 'nn_pos_opp4_cum', 'nn_neg_opp4_cum',
            'nn_pos_same8_cum', 'nn_neg_same8_cum', 'nn_pos_opp8_cum', 'nn_neg_opp8_cum',
            'kk', 'pp', 'nn', 'pn']
            
    samples = {k: [] for k in keys}
    
    # We use 1000 realizations as default per MakeCovs script
    count = 0
    # Search for existing files to speed up or limit
    # To save time, we can try to load fewer if just for plotting, but KL needs good stats.
    # The original script uses all. I will follow suit but maybe add a check.
    
    for i in tqdm(range(n_files), desc="Loading SLICS"):
        for subfield in range(4):
            try:
                filepath = f"{slics_dir}/SLICS_stats_s467_th_morebins_{i}_subfield{subfield}.npz"
                if not os.path.exists(filepath):
                    continue
                
                with np.load(filepath) as data:
                    # Special handling for kappa truncation
                    if 'kappa' in data:
                        samples['kappa'].append(data['kappa'])
                    
                    # Direct mapping for others
                    for k in keys:
                        if k == 'kappa': continue
                        if k in data:
                            samples[k].append(data[k])
                count += 1
            except Exception as e:
                # print(f"Error loading file {filepath}: {e}")
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
    sigma[sigma == 0] = 1.0
    mask = np.abs(fid_vec) > sigma_thresh * sigma
    
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
        
        mask = mask & (kl_values <= 0.1)
        
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
    for i in range(corr.shape[0]):
        if not corr_mask[i]: continue
        for j in range(i + 1, corr.shape[0]):
            if not corr_mask[j]: continue
            if np.abs(corr[i, j]) >= 0.9999:
                # Remove the bin with lower S/N
                # mapped back to original indices
                idx_i = np.where(mask)[0][i]
                idx_j = np.where(mask)[0][j]
                
                s_i = sigma[idx_i] if sigma[idx_i] > 0 else 1.0
                s_j = sigma[idx_j] if sigma[idx_j] > 0 else 1.0
                
                sn_i = np.abs(fid_vec[idx_i]) / s_i
                sn_j = np.abs(fid_vec[idx_j]) / s_j
                
                if sn_i <= sn_j:
                    corr_mask[i] = False
                else:
                    corr_mask[j] = False

    # Apply correlation mask to the S/N mask
    if corr_mask.sum() < len(corr_mask):
         # Get indices of S/N passed bins
        sn_indices = np.where(mask)[0]
        # Keep only those that passed correlation
        final_indices = sn_indices[corr_mask]
        
        new_mask = np.zeros_like(mask)
        new_mask[final_indices] = True
        return new_mask

    return mask

def get_component_structure(name):
    return []

def plot_corr_matrix(name, cov, mask, filename, data_slics, area_scaling):
    # Plot FULL matrix
    sigma = np.sqrt(np.diag(cov))
    sigma[sigma == 0] = 1.0
    corr = cov / np.outer(sigma, sigma)
    
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    
    # Plot using Seaborn Heatmap with linear scale
    # cbar=False because we will add it manually to ensure height match
    sns.heatmap(corr, ax=ax, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, cbar=False, xticklabels=False, yticklabels=False)

    # Highlight zero covariance points
    zero_mask = np.abs(cov) < 1e-10
    if np.any(zero_mask):
        sns.heatmap(np.zeros_like(corr), ax=ax, mask=~zero_mask, cbar=False,
                    cmap=ListedColormap(['#d9d9d9']), vmin=-1, vmax=1,
                    xticklabels=False, yticklabels=False, square=True)

    # Add black border to the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    # Create matched colorbar (thinner)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    
    # Add colorbar
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='Correlation Coefficient')
    
    # Ensure limits are correct after axes modifications
    ax.set_xlim(0, corr.shape[1])
    ax.set_ylim(corr.shape[0], 0)
    
    # Hierarchy for Labels and Black Dashed Lines
    hierarchy = {
        'kappa_all': ['kappa_pos', 'kappa_neg'],
        'nn_1': ['nn_pos_same1', 'nn_neg_same1', 'nn_pos_opp1', 'nn_neg_opp1'],
        'nn_2': ['nn_pos_same2', 'nn_neg_same2', 'nn_pos_opp2', 'nn_neg_opp2'],
        'nn_4': ['nn_pos_same4', 'nn_neg_same4', 'nn_pos_opp4', 'nn_neg_opp4'],
        'nn_8': ['nn_pos_same8', 'nn_neg_same8', 'nn_pos_opp8', 'nn_neg_opp8'],
        'nn_all': ['nn_1', 'nn_2', 'nn_4', 'nn_8'],
        # 'all' is typically defects only (kappa_all + nn_all)
        'all': ['kappa_all', 'nn_all'],
        'all_and_pdf': ['all', 'kappa'],
        # Structure requested: D-CDF, kNN, PDF, kappa-2PCF
        # Original: all (dcdp+knn), kappa (pdf), kk
        # So we break 'all' into 'kappa_all' (D-CDF) and 'nn_all' (kNN)
        'all_howls_and_kk_custom': ['kappa_all', 'nn_all', 'kappa', 'kk'],
        'all_howls_and_kk': ['all', 'kappa', 'kk'], 
        'all_and_kk': ['all', 'kk'],
        'dcorr': ['pp', 'nn', 'pn']
    }
    
    # Label Map
    label_map = {
        'kappa_all': 'D-CDF',
        'nn_all': 'kNN',
        'kappa': 'PDF',
        'kk': r'$\kappa$-2PCF'
    }

    def get_leaves(key):
        # Special handling for custom top-level structure
        if key == 'all_howls_and_kk':
            # Use custom flattening for this specific key
            return get_leaves('all_howls_and_kk_custom')

        if key not in hierarchy:
            if f"{key}_mean" in data_slics:
                # Use mapped label if available, else key
                label = label_map.get(key, key)
                return [(label, len(data_slics[f"{key}_mean"]))]
            else:
                return []
        
        leaves = []
        for child in hierarchy[key]:
            # Recursion
            # If the child itself has a Label Map entry, treat it as a leaf for labeling purposes?
            # Or do we want to break it down further?
            # The user asked for "D-CDF, kNN, PDF, kappa-2PCF".
            # D-CDF is kappa_all. kNN is nn_all.
            # So if we encounter these keys, we should STOP recursing and treat them as blocks.
            
            if child in label_map and key == 'all_howls_and_kk_custom':
                # Treat as leaf block using pre-calculated size
                # We need to compute total length of this block
                length = 0
                
                # Helper to get size of any node
                def get_size(node):
                     if node not in hierarchy:
                         return len(data_slics.get(f"{node}_mean", []))
                     s = 0
                     for c in hierarchy[node]:
                         s += get_size(c)
                     return s
                
                length = get_size(child)
                leaves.append((label_map[child], length))
            else:
                leaves.extend(get_leaves(child))
        return leaves
    
    # Custom logic for 'all_howls_and_kk' to ensure correct breakdown
    if name == 'all_howls_and_kk':
        # Manually define the leaves based on user request "D-CDF, kNN, PDF, kappa-2PCF"
        # We need to know the lengths of D-CDF (kappa_all), kNN (nn_all), PDF (kappa), 2PCF (kk)
        # We can fetch them recursively or directly if keys exist
        
        def get_len_recursive(k):
            if f"{k}_mean" in data_slics:
                return len(data_slics[f"{k}_mean"])
            # Fallback for composite keys not in data_slics directly?
            # 'kappa_all', 'nn_all' are constructed by us if not in file, but we added logic to skip if missing.
            # Assuming they exist or can be summed up.
            if k in hierarchy:
                total_len = 0
                for child in hierarchy[k]:
                    total_len += get_len_recursive(child)
                return total_len
            return 0
            
        len_dcdf = get_len_recursive('kappa_all')
        len_knn = get_len_recursive('nn_all')
        len_pdf = get_len_recursive('kappa')
        len_kk = get_len_recursive('kk')
        
        leaves = [
            ('D-CDF', len_dcdf),
            ('kNN', len_knn),
            ('PDF', len_pdf),
            (r'$\kappa$-2PCF', len_kk)
        ]
    else:
        leaves = get_leaves(name)
        if not leaves:
            if f"{name}_mean" in data_slics:
                leaves = [(name, len(data_slics[f"{name}_mean"]))]
    
    # Map original lengths (Unfiltered)
    current_idx = 0
    boundaries = []
    labels = []
    label_positions = []
    
    total_len_leaves = sum([L for _, L in leaves])
    print(f"\n--- Data Component Proportions for {name} ---")
    
    for leaf_name, leaf_len in leaves:
        start = current_idx
        end = current_idx + leaf_len
        
        boundaries.append(end)
        label_pos = start + leaf_len / 2.0
        labels.append(leaf_name)
        label_positions.append(label_pos)
        
        ratio_kept = np.sum(mask[start:end]) / leaf_len * 100
        print(f"{leaf_name}: {leaf_len} total bins, {np.sum(mask[start:end])} kept ({ratio_kept:.2f}%)")
        
        current_idx = end
    
    # Draw Black Dashed Lines (Separators)
    if len(boundaries) > 0:
        for b in boundaries[:-1]:
            pos = b - 0.5
            ax.axvline(x=pos, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
            ax.axhline(y=pos, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    
    # Highlight Kept Bins with Red Boxes
    # mask is boolean array of same length as vector
    # Find contiguous true segments
    kept_indices = np.where(mask)[0]
    if len(kept_indices) > 0:
        segments = []
        start_seg = kept_indices[0]
        curr_seg = kept_indices[0]
        for idx in kept_indices[1:]:
            if idx == curr_seg + 1:
                curr_seg = idx
            else:
                segments.append((start_seg, curr_seg))
                start_seg = idx
                curr_seg = idx
        segments.append((start_seg, curr_seg))
        
        for start, end in segments:
            # Draw box on diagonal
            # (col, row). 
            # Top-left of matrix is (0,0).
            # Box extends from start-0.5 to end+0.5
            width = end - start + 1
            xy = (start - 0.5, start - 0.5)
            
            rect = plt.Rectangle(xy, width, width, 
                                 fill=False, edgecolor='black', lw=1.0, linestyle='-', zorder=10)
            ax.add_patch(rect)

    # Remove standard ticks
    ax.tick_params(axis='both', which='both', length=0)
    
    # Add custom labels
    ax.set_xticks(label_positions)
    ax.set_xticklabels(labels, rotation=90, ha='center')
    ax.xaxis.tick_top()
    
    ax.set_yticks(label_positions)
    ax.set_yticklabels(labels, va='center')
    
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    print("Loading data...")
    # Load Main Data
    try:
        data_slics = dict(np.load(f"{SLICS_DIR}/SLICS_stats_cov_mean_s467_th_morebins_subfields.npz"))
    except FileNotFoundError:
        print(f"Could not find data file at {SLICS_DIR}")
        exit(1)
        
    # Load SLICS samples for KL calculation
    slics_samples = load_slics_samples(SLICS_DIR)

    # Construct unified `all_howls_and_kk` samples as per Fisher script
    # We need to manually construct the sample arrays for the target key
    # slics_samples dict has base components.
    
    # Constructing 'all_howls_and_kk' samples
    # Structure: [all, kappa, kk] where all = [kappa_all, nn_all]
    # kappa_all = [kappa_pos, kappa_neg], nn_all = [nn_1, nn_2, nn_4, nn_8]
    
    # We need to construct these composites if they are missing
    if 'kappa_all' not in slics_samples:
        slics_samples['kappa_all'] = np.hstack([slics_samples['kappa_pos'], slics_samples['kappa_neg']])
    if 'nn_1' not in slics_samples:
        slics_samples['nn_1'] = np.hstack([slics_samples['nn_pos_same1'], slics_samples['nn_neg_same1'], 
                                           slics_samples['nn_pos_opp1'], slics_samples['nn_neg_opp1']])
    if 'nn_2' not in slics_samples:
        slics_samples['nn_2'] = np.hstack([slics_samples['nn_pos_same2'], slics_samples['nn_neg_same2'], 
                                           slics_samples['nn_pos_opp2'], slics_samples['nn_neg_opp2']])
    if 'nn_4' not in slics_samples:
        slics_samples['nn_4'] = np.hstack([slics_samples['nn_pos_same4'], slics_samples['nn_neg_same4'], 
                                           slics_samples['nn_pos_opp4'], slics_samples['nn_neg_opp4']])
    if 'nn_8' not in slics_samples:
        slics_samples['nn_8'] = np.hstack([slics_samples['nn_pos_same8'], slics_samples['nn_neg_same8'], 
                                           slics_samples['nn_pos_opp8'], slics_samples['nn_neg_opp8']])
    if 'nn_all' not in slics_samples:
        slics_samples['nn_all'] = np.hstack([slics_samples['nn_1'], slics_samples['nn_2'], 
                                             slics_samples['nn_4'], slics_samples['nn_8']])
    if 'all' not in slics_samples:
        slics_samples['all'] = np.hstack([slics_samples['kappa_all'], slics_samples['nn_all']])
    if 'all_howls_and_kk' not in slics_samples:
        slics_samples['all_howls_and_kk'] = np.hstack([slics_samples['all'], slics_samples['kappa'], slics_samples['kk']])
    
    targets = [
        'all_howls_and_kk'
    ]
    
    area_scaling = 25 / 15000 

    for key in targets:
        print(f"Processing {key}...")
        mean_key = f"{key}_mean"
        cov_key = f"{key}_cov"
        
        mean_vec = None
        cov_vec = None

        if mean_key in data_slics and cov_key in data_slics:
            mean_vec = data_slics[mean_key]
            cov_vec = data_slics[cov_key] * area_scaling
        else:
            # Need to construct it if it's missing (some composites might not be pre-calculated)
            # Actually File 6 suggests they are pre-calculated in data_slics for all combinations used in Fisher.
            print(f"Skipping {key} (not found in data)")
            continue
            
        samples = slics_samples.get(key)
        if samples is None:
             print(f"Warning: Samples for {key} not found. Skipping KL filter.")
        
        mask = get_kept_mask(key, mean_vec, cov_vec, samples=samples)
        
        plot_corr_matrix(key, cov_vec, mask, f"{OUT_DIR}/{key}_corr.png", data_slics, area_scaling)
