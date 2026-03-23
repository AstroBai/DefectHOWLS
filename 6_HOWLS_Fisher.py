from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

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

def get_bin_coords(name):
    # Bin definitions provided by user
    bin_edges_kappa = np.linspace(-0.05, 0.05, 100)
    bin_edges1 = np.logspace(np.log10(0.59), np.log10(100), 100)
    bin_edges2 = np.logspace(np.log10(0.59), np.log10(100), 100)
    bin_edges4 = np.logspace(np.log10(0.59), np.log10(100), 100)
    bin_edges8 = np.logspace(np.log10(0.59), np.log10(100), 100)
    
    if name == "kappa":
        centers = (bin_edges_kappa[:-1] + bin_edges_kappa[1:]) / 2
        return centers# Adjust for slice
    
    # Cumulative stats use bin edges
    if name in ["kappa_pos", "kappa_neg"]:
        return bin_edges_kappa
        
    if "nn_" in name:
        if "1" in name: return bin_edges1
        if "2" in name: return bin_edges2
        if "4" in name: return bin_edges4
        if "8" in name: return bin_edges8
        
    if name == "kk":
        # Use theta from data_slics if available, otherwise approximation
        try:
            # Access global data_slics if available
            return data_slics['theta']
        except:
             # Fallback if global not ready (though it should be called after load)
             return np.logspace(np.log10(0.59), np.log10(100), 100)
             
    return None

def plot_ridgeline_kl(data_matrix, mask, title, filename_suffix):
    """
    Plots vertical ridgeline plot with KL annotations for kept bins.
    """
    if mask.sum() == 0:
        return

    kept_indices = np.where(mask)[0]
    selected_data = data_matrix[:, kept_indices]
    n_plot_bins = len(kept_indices)
    
    # Vertical layout
    fig, ax = plt.subplots(figsize=(max(6, n_plot_bins * 0.4), 8))
    
    overlap = 0.5
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot_bins))

    for i, bin_idx in enumerate(kept_indices):
        x_data = selected_data[:, i]
        
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
        
        ax.text(baseline + 0.1, y_grid.min(), f"KL={kl:.3f}", rotation=90, va='top', fontsize=7, zorder=n_plot_bins-i+1)

    ax.set_xticks(np.arange(len(kept_indices)))
    
    # Generate labels based on physical coordinates
    coords = get_bin_coords(filename_suffix)
    labels = []
    
    for idx in kept_indices:
        if coords is not None and idx < len(coords):
            if filename_suffix == "kk": # Log scale for theta usually
                 labels.append(f"{coords[idx]:.1f}")
            elif "kappa" in filename_suffix and "pos" not in filename_suffix and "neg" not in filename_suffix:
                 labels.append(f"{coords[idx]:.4f}")
            else:
                 labels.append(f"{coords[idx]:.1f}")
        else:
            labels.append(f"Bin {idx}")
            
    ax.set_xticklabels(labels, rotation=90)
    ax.set_title(title)
    if "nn_" in filename_suffix:
        ax.set_xlabel("Scale [arcmin]")
    elif filename_suffix == "kk":
        ax.set_xlabel("Theta [arcmin]")
    elif "kappa" in filename_suffix:
        ax.set_xlabel("Kappa")
        
    ax.set_ylabel("Value")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs("outfiles/Fisher_Ridgelines", exist_ok=True)
    plt.savefig(f"outfiles/Fisher_Ridgelines/{filename_suffix}.png", dpi=300)
    plt.close()

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

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
    for i in tqdm(range(n_files), desc="Loading SLICS"):
        for subfield in range(4):
            try:
                filepath = f"{slics_dir}/SLICS_stats_s467_th_{i}_subfield{subfield}.npz"
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
                print(f"Error loading file {filepath}: {e}")
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

n_realizations = 924 * 4

def filter_data_block(name, fid_vec, deriv, cov, n_realizations, samples=None, sigma_thresh=2.0):
    # First filter: remove bins with low signal-to-noise
    sigma = np.sqrt(np.diag(cov))
    mask = np.abs(fid_vec) > sigma_thresh * sigma
    
    # KL Filter
    if samples is not None:
        kl_values = np.zeros(fid_vec.shape[0])
        # Calculate KL only for S/N passed bins
        for i in range(fid_vec.shape[0]):
             if mask[i]:
                 kl_values[i] = calculate_kl(samples[:, i])
        
        original_count = mask.sum()
        mask = mask & (kl_values <= 0.1)
        kl_removed_count = original_count - mask.sum()
        if kl_removed_count > 0:
            print(f"{name}: removed {kl_removed_count} bins due to KL > 0.1")

    if mask.sum() == 0:
        # Fallback: keep all if none survive? Or raise error?
        # If strict, raise error. But for robustness, maybe warn.
        # User saw "no bins survive" mostly due to double filtering. 
        # With clean filtering, if truly no bins survive, it's a data issue.
        # However, to be safe against crash, let's keep S/N ones if KL fails all?
        # sticking to original logic:
        pass # raise ValueError below if still 0
        
    if mask.sum() == 0:
        print(f"WARNING: {name}: no bins survive S/N + KL cuts. Using all bins as fallback (dangerous).")
        mask[:] = True

    # Apply first cut
    cov_trim = cov[np.ix_(mask, mask)]
    
    # Second filter: remove bins with 100% correlation
    sigma_trim = np.sqrt(np.diag(cov_trim))
    # Avoid div by zero
    sigma_trim[sigma_trim == 0] = 1.0
    
    corr = cov_trim / np.outer(sigma_trim, sigma_trim)
    
    # Find pairs with |correlation| >= 0.9999
    corr_mask = np.ones(corr.shape[0], dtype=bool)
    for i in range(corr.shape[0]):
        if not corr_mask[i]: continue
        for j in range(i + 1, corr.shape[0]):
            if not corr_mask[j]: continue
            if np.abs(corr[i, j]) >= 0.9999:
                # Remove the bin with lower S/N
                idx_i = np.where(mask)[0][i]
                idx_j = np.where(mask)[0][j]
                
                # Check sigma != 0
                s_i = sigma[idx_i] if sigma[idx_i] > 0 else 1.0
                s_j = sigma[idx_j] if sigma[idx_j] > 0 else 1.0
                
                sn_i = np.abs(fid_vec[idx_i]) / s_i
                sn_j = np.abs(fid_vec[idx_j]) / s_j
                
                if sn_i <= sn_j:
                    corr_mask[i] = False
                else:
                    corr_mask[j] = False
    
    if corr_mask.sum() == 0:
        print(f"WARNING: {name}: no bins survive Correlation cuts. Using previous mask.")
    else:
        # Update mask
        original_indices = np.where(mask)[0]
        final_indices = original_indices[corr_mask]
        mask = np.zeros_like(mask)
        mask[final_indices] = True
    
    kept_indices = np.where(mask)[0]
    
    # Plot Ridgelines
    if samples is not None:
        plot_ridgeline_kl(samples, mask, f"{name} Distribution", name)

    print(f"{name} final kept bins: {kept_indices.tolist()} (Count: {len(kept_indices)})")
    
    # Apply final mask
    cov_final = cov[np.ix_(mask, mask)]
    # Hartlap
    hartlap = (n_realizations - mask.sum() - 2) / (n_realizations - 1)
    if mask.sum() >= n_realizations - 2:
        print(f"WARNING: {name}: too many bins ({mask.sum()}) for Hartlap correction (N={n_realizations}). InvCov might be unstable.")
        hartlap = 1.0 
        
    try:
        inv_cov = np.linalg.inv(cov_final) * hartlap
    except np.linalg.LinAlgError:
        print(f"ERROR: {name}: Singular matrix. Using pseudo-inverse.")
        inv_cov = np.linalg.pinv(cov_final) * hartlap

    return deriv[:, mask], cov_final, inv_cov, mask

class FisherDataBlock:
    def __init__(self, name, deriv, cov, n_realizations, fid_vec=None, samples=None, mask=None, inv_cov=None, final_cov=None, final_deriv=None):
        self.name = name
        self.deriv = deriv
        self.cov = cov
        self.n = n_realizations
        self.samples = samples
        self.fid_vec = fid_vec
        self.mask = mask
        self.inv_cov = inv_cov
        self.final_cov = final_cov
        self.final_deriv = final_deriv

    def filter(self):
        # Wrapper for filter_data_block
        fd, fc, ic, m = filter_data_block(self.name, self.fid_vec, self.deriv, self.cov, self.n, samples=self.samples)
        self.final_deriv = fd
        self.final_cov = fc
        self.inv_cov = ic
        self.mask = m
        return self

    @classmethod
    def combine(cls, name, components, combined_deriv, combined_cov, n_realizations, fid_vec=None, samples=None, sigma_thresh=2.0):
        if fid_vec is None:
            if samples is not None and len(samples) > 0:
                fid_vec = np.mean(samples, axis=0)
            else:
                raise ValueError(f"{name}: fid_vec is required for combined filtering.")

        deriv_final, cov_final, inv_cov, full_mask = filter_data_block(
            name,
            fid_vec,
            combined_deriv,
            combined_cov,
            n_realizations,
            samples=samples,
            sigma_thresh=sigma_thresh,
        )

        print(f"{name}: Combined mask kept {full_mask.sum()} / {len(full_mask)} bins")

        return cls(
            name,
            combined_deriv,
            combined_cov,
            n_realizations,
            fid_vec=fid_vec,
            samples=samples,
            mask=full_mask,
            inv_cov=inv_cov,
            final_cov=cov_final,
            final_deriv=deriv_final,
        )

def create_block(name, key_mean, key_deriv, key_cov, key_samples):
    return FisherDataBlock(name, key_deriv, key_cov, n_realizations, fid_vec=key_mean, samples=key_samples)

data_om02 = np.load(f"{OM02_DIR}/OM02_stats_cov_mean_s467_th.npz")
data_om04 = np.load(f"{OM04_DIR}/OM04_stats_cov_mean_s467_th.npz")
data_s807 = np.load(f"{S807_DIR}/S807_stats_cov_mean_s467_th.npz")
data_s809 = np.load(f"{S809_DIR}/S809_stats_cov_mean_s467_th.npz")
data_slics = np.load(f"{SLICS_DIR}/SLICS_stats_cov_mean_s467_th_subfields.npz")

slics_samples = load_slics_samples(SLICS_DIR)

# Construct composite samples
slics_samples['kappa_all'] = np.hstack([slics_samples['kappa_pos'], slics_samples['kappa_neg']])
slics_samples['nn_1'] = np.hstack([slics_samples['nn_pos_same1'], slics_samples['nn_neg_same1'], 
                                   slics_samples['nn_pos_opp1'], slics_samples['nn_neg_opp1']])
slics_samples['nn_2'] = np.hstack([slics_samples['nn_pos_same2'], slics_samples['nn_neg_same2'], 
                                   slics_samples['nn_pos_opp2'], slics_samples['nn_neg_opp2']])
slics_samples['nn_4'] = np.hstack([slics_samples['nn_pos_same4'], slics_samples['nn_neg_same4'], 
                                   slics_samples['nn_pos_opp4'], slics_samples['nn_neg_opp4']])
slics_samples['nn_8'] = np.hstack([slics_samples['nn_pos_same8'], slics_samples['nn_neg_same8'], 
                                   slics_samples['nn_pos_opp8'], slics_samples['nn_neg_opp8']])
slics_samples['nn_all'] = np.hstack([slics_samples['nn_1'], slics_samples['nn_2'], 
                                     slics_samples['nn_4'], slics_samples['nn_8']])
slics_samples['all'] = np.hstack([slics_samples['kappa_all'], slics_samples['nn_all']])
slics_samples['all_and_pdf'] = np.hstack([slics_samples['all'], slics_samples['kappa']])
slics_samples['all_howls_and_kk'] = np.hstack([slics_samples['all'], slics_samples['kappa'], slics_samples['kk']])
slics_samples['all_and_kk'] = np.hstack([slics_samples['all'], slics_samples['kk']])
slics_samples['dcorr'] = np.hstack([slics_samples['pp'], slics_samples['nn'], slics_samples['pn']])

params = np.array([[0.2000, 0.842],
                  [0.4000, 0.842],
                  [0.3134, 0.707],
                  [0.3134, 0.977]])
params_fid = np.array([0.3134, 0.842])

kappa_deriv = []
kappa_pos_deriv = []
kappa_neg_deriv = []
nn_pos_same1_deriv = []
nn_neg_same1_deriv = []
nn_pos_opp1_deriv = []
nn_neg_opp1_deriv = []
nn_pos_same2_deriv = []
nn_neg_same2_deriv = []
nn_pos_opp2_deriv = []
nn_neg_opp2_deriv = []
nn_pos_same4_deriv = []
nn_neg_same4_deriv = []
nn_pos_opp4_deriv = []
nn_neg_opp4_deriv = []
nn_pos_same8_deriv = []
nn_neg_same8_deriv = []
nn_pos_opp8_deriv = []
nn_neg_opp8_deriv = [] 
kappa_all_deriv = []
nn_1_deriv = []
nn_2_deriv = []
nn_4_deriv = []
nn_8_deriv = []
nn_all_deriv = []
all_deriv = []
kk_deriv = []
pp_deriv = []
nn_deriv = []
pn_deriv = []
dcorr_deriv = []
all_and_pdf_deriv = []
all_howls_and_kk_deriv = []
all_and_kk_deriv = []
for i in range(2):
    if i == 0:
        delta_p = params[1,0] - params[0,0]
        data_plus = data_om04
        data_minus = data_om02
    else:
        delta_p = params[3,1] - params[2,1]
        data_plus = data_s809
        data_minus = data_s807

    kappa_deriv.append((data_plus['kappa_mean'] - data_minus['kappa_mean']) / delta_p)
    kappa_pos_deriv.append((data_plus['kappa_pos_mean'] - data_minus['kappa_pos_mean']) / delta_p)
    kappa_neg_deriv.append((data_plus['kappa_neg_mean'] - data_minus['kappa_neg_mean']) / delta_p)
    nn_pos_same1_deriv.append((data_plus['nn_pos_same1_mean'] - data_minus['nn_pos_same1_mean']) / delta_p)
    nn_neg_same1_deriv.append((data_plus['nn_neg_same1_mean'] - data_minus['nn_neg_same1_mean']) / delta_p)
    nn_pos_opp1_deriv.append((data_plus['nn_pos_opp1_mean'] - data_minus['nn_pos_opp1_mean']) / delta_p)
    nn_neg_opp1_deriv.append((data_plus['nn_neg_opp1_mean'] - data_minus['nn_neg_opp1_mean']) / delta_p)
    nn_pos_same2_deriv.append((data_plus['nn_pos_same2_mean'] - data_minus['nn_pos_same2_mean']) / delta_p)
    nn_neg_same2_deriv.append((data_plus['nn_neg_same2_mean'] - data_minus['nn_neg_same2_mean']) / delta_p)
    nn_pos_opp2_deriv.append((data_plus['nn_pos_opp2_mean'] - data_minus['nn_pos_opp2_mean']) / delta_p)
    nn_neg_opp2_deriv.append((data_plus['nn_neg_opp2_mean'] - data_minus['nn_neg_opp2_mean']) / delta_p)
    nn_pos_same4_deriv.append((data_plus['nn_pos_same4_mean'] - data_minus['nn_pos_same4_mean']) / delta_p)
    nn_neg_same4_deriv.append((data_plus['nn_neg_same4_mean'] - data_minus['nn_neg_same4_mean']) / delta_p)
    nn_pos_opp4_deriv.append((data_plus['nn_pos_opp4_mean'] - data_minus['nn_pos_opp4_mean']) / delta_p)
    nn_neg_opp4_deriv.append((data_plus['nn_neg_opp4_mean'] - data_minus['nn_neg_opp4_mean']) / delta_p)
    nn_pos_same8_deriv.append((data_plus['nn_pos_same8_mean'] - data_minus['nn_pos_same8_mean']) / delta_p)
    nn_neg_same8_deriv.append((data_plus['nn_neg_same8_mean'] - data_minus['nn_neg_same8_mean']) / delta_p)
    nn_pos_opp8_deriv.append((data_plus['nn_pos_opp8_mean'] - data_minus['nn_pos_opp8_mean']) / delta_p)
    nn_neg_opp8_deriv.append((data_plus['nn_neg_opp8_mean'] - data_minus['nn_neg_opp8_mean']) / delta_p)    
    kappa_all_deriv.append((data_plus['kappa_all_mean'] - data_minus['kappa_all_mean']) / delta_p)
    nn_1_deriv.append((data_plus['nn_1_mean'] - data_minus['nn_1_mean']) / delta_p)
    nn_2_deriv.append((data_plus['nn_2_mean'] - data_minus['nn_2_mean']) / delta_p)
    nn_4_deriv.append((data_plus['nn_4_mean'] - data_minus['nn_4_mean']) / delta_p)
    nn_8_deriv.append((data_plus['nn_8_mean'] - data_minus['nn_8_mean']) / delta_p)
    nn_all_deriv.append((data_plus['nn_all_mean'] - data_minus['nn_all_mean']) / delta_p)
    all_deriv.append((data_plus['all_mean'] - data_minus['all_mean']) / delta_p)
    kk_deriv.append((data_plus['kk_mean'] - data_minus['kk_mean']) / delta_p)
    all_and_pdf_deriv.append((data_plus['all_and_pdf_mean'] - data_minus['all_and_pdf_mean']) / delta_p)
    all_howls_and_kk_deriv.append((data_plus['all_howls_and_kk_mean'] - data_minus['all_howls_and_kk_mean']) / delta_p)
    all_and_kk_deriv.append((data_plus['all_and_kk_mean'] - data_minus['all_and_kk_mean']) / delta_p)
    pp_deriv.append((data_plus['pp_mean'] - data_minus['pp_mean']) / delta_p)
    nn_deriv.append((data_plus['nn_mean'] - data_minus['nn_mean']) / delta_p)
    pn_deriv.append((data_plus['pn_mean'] - data_minus['pn_mean']) / delta_p)
    dcorr_deriv.append((data_plus['dcorr_mean'] - data_minus['dcorr_mean']) / delta_p)

kappa_deriv = np.array(kappa_deriv)
kappa_pos_deriv = np.array(kappa_pos_deriv)
kappa_neg_deriv = np.array(kappa_neg_deriv)
nn_pos_same1_deriv = np.array(nn_pos_same1_deriv)
nn_neg_same1_deriv = np.array(nn_neg_same1_deriv)
nn_pos_opp1_deriv = np.array(nn_pos_opp1_deriv)
nn_neg_opp1_deriv = np.array(nn_neg_opp1_deriv)
nn_pos_same2_deriv = np.array(nn_pos_same2_deriv)
nn_neg_same2_deriv = np.array(nn_neg_same2_deriv)
nn_pos_opp2_deriv = np.array(nn_pos_opp2_deriv)
nn_neg_opp2_deriv = np.array(nn_neg_opp2_deriv)
nn_pos_same4_deriv = np.array(nn_pos_same4_deriv)
nn_neg_same4_deriv = np.array(nn_neg_same4_deriv)
nn_pos_opp4_deriv = np.array(nn_pos_opp4_deriv)
nn_neg_opp4_deriv = np.array(nn_neg_opp4_deriv)
nn_pos_same8_deriv = np.array(nn_pos_same8_deriv)
nn_neg_same8_deriv = np.array(nn_neg_same8_deriv)
nn_pos_opp8_deriv = np.array(nn_pos_opp8_deriv)
nn_neg_opp8_deriv = np.array(nn_neg_opp8_deriv) 
kappa_all_deriv = np.array(kappa_all_deriv)
nn_1_deriv = np.array(nn_1_deriv)
nn_2_deriv = np.array(nn_2_deriv)
nn_4_deriv = np.array(nn_4_deriv)
nn_8_deriv = np.array(nn_8_deriv)
nn_all_deriv = np.array(nn_all_deriv)
all_deriv = np.array(all_deriv) 
kk_deriv = np.array(kk_deriv)
all_and_pdf_deriv = np.array(all_and_pdf_deriv)
all_howls_and_kk_deriv = np.array(all_howls_and_kk_deriv)
all_and_kk_deriv = np.array(all_and_kk_deriv)
pp_deriv = np.array(pp_deriv)
nn_deriv = np.array(nn_deriv)
pn_deriv = np.array(pn_deriv)
dcorr_deriv = np.array(dcorr_deriv)

area_scaling = 25 / 15000   # euclid
kappa_cov = data_slics['kappa_cov'] * area_scaling
kappa_pos_cov = data_slics['kappa_pos_cov'] * area_scaling
kappa_neg_cov = data_slics['kappa_neg_cov'] * area_scaling
nn_pos_same1_cov = data_slics['nn_pos_same1_cov'] * area_scaling
nn_neg_same1_cov = data_slics['nn_neg_same1_cov'] * area_scaling
nn_pos_opp1_cov = data_slics['nn_pos_opp1_cov'] * area_scaling
nn_neg_opp1_cov = data_slics['nn_neg_opp1_cov'] * area_scaling
nn_pos_same2_cov = data_slics['nn_pos_same2_cov'] * area_scaling
nn_neg_same2_cov = data_slics['nn_neg_same2_cov'] * area_scaling
nn_pos_opp2_cov = data_slics['nn_pos_opp2_cov'] * area_scaling
nn_neg_opp2_cov = data_slics['nn_neg_opp2_cov'] * area_scaling
nn_pos_same4_cov = data_slics['nn_pos_same4_cov'] * area_scaling
nn_neg_same4_cov = data_slics['nn_neg_same4_cov'] * area_scaling
nn_pos_opp4_cov = data_slics['nn_pos_opp4_cov'] * area_scaling
nn_neg_opp4_cov = data_slics['nn_neg_opp4_cov'] * area_scaling
nn_pos_same8_cov = data_slics['nn_pos_same8_cov'] * area_scaling
nn_neg_same8_cov = data_slics['nn_neg_same8_cov'] * area_scaling
nn_pos_opp8_cov = data_slics['nn_pos_opp8_cov'] * area_scaling
nn_neg_opp8_cov = data_slics['nn_neg_opp8_cov'] * area_scaling
kappa_all_cov = data_slics['kappa_all_cov'] * area_scaling
nn_1_cov = data_slics['nn_1_cov'] * area_scaling
nn_2_cov = data_slics['nn_2_cov'] * area_scaling
nn_4_cov = data_slics['nn_4_cov'] * area_scaling
nn_8_cov = data_slics['nn_8_cov'] * area_scaling
nn_all_cov = data_slics['nn_all_cov'] * area_scaling
all_cov = data_slics['all_cov'] * area_scaling
kk_cov = data_slics['kk_cov'] * area_scaling
all_and_pdf_cov = data_slics['all_and_pdf_cov'] * area_scaling  
all_howls_and_kk_cov = data_slics['all_howls_and_kk_cov'] * area_scaling
all_and_kk_cov = data_slics['all_and_kk_cov'] * area_scaling
pp_cov = data_slics['pp_cov'] * area_scaling
nn_cov = data_slics['nn_cov'] * area_scaling
pn_cov = data_slics['pn_cov'] * area_scaling
dcorr_cov = data_slics['dcorr_cov'] * area_scaling
n = n_realizations

# 1. Initialize Individual Blocks
kappa_block = create_block("kappa", data_slics["kappa_mean"], kappa_deriv, kappa_cov, slics_samples.get("kappa")).filter()
kappa_pos_block = create_block("kappa_pos", data_slics["kappa_pos_mean"], kappa_pos_deriv, kappa_pos_cov, slics_samples.get("kappa_pos")).filter()
kappa_neg_block = create_block("kappa_neg", data_slics["kappa_neg_mean"], kappa_neg_deriv, kappa_neg_cov, slics_samples.get("kappa_neg")).filter()

nn_blocks = {}
for scale in [1, 2, 4, 8]:
    for t in ['pos_same', 'neg_same', 'pos_opp', 'neg_opp']:
        key = f"nn_{t}{scale}"
        # We need to access the variables like nn_pos_same1_deriv etc.
        # Since they are local variables, we assume they are available.
        # Using locals() or globals()? In a script, globals() works.
        d_val = globals()[f"{key}_deriv"]
        c_val = globals()[f"{key}_cov"]
        m_val = data_slics[f"{key}_mean"]
        s_val = slics_samples.get(key)
        
        block = create_block(key, m_val, d_val, c_val, s_val).filter()
        nn_blocks[key] = block

kk_block = create_block("kk", data_slics["kk_mean"], kk_deriv, kk_cov, slics_samples.get("kk")).filter()
pp_block = create_block("pp", data_slics["pp_mean"], pp_deriv, pp_cov, slics_samples.get("pp")).filter()
nn_block = create_block("nn", data_slics["nn_mean"], nn_deriv, nn_cov, slics_samples.get("nn")).filter()
pn_block = create_block("pn", data_slics["pn_mean"], pn_deriv, pn_cov, slics_samples.get("pn")).filter()

# Assign back to variables for compatibility if needed (optional, but good for safety)
kappa_deriv, kappa_cov, kappa_inv_cov, kappa_mask = kappa_block.final_deriv, kappa_block.final_cov, kappa_block.inv_cov, kappa_block.mask
kappa_pos_deriv, kappa_pos_cov, kappa_pos_inv_cov, kappa_pos_mask = kappa_pos_block.final_deriv, kappa_pos_block.final_cov, kappa_pos_block.inv_cov, kappa_pos_block.mask
kappa_neg_deriv, kappa_neg_cov, kappa_neg_inv_cov, kappa_neg_mask = kappa_neg_block.final_deriv, kappa_neg_block.final_cov, kappa_neg_block.inv_cov, kappa_neg_block.mask
kk_deriv, kk_cov, kk_inv_cov, kk_mask = kk_block.final_deriv, kk_block.final_cov, kk_block.inv_cov, kk_block.mask
pp_deriv, pp_cov, pp_inv_cov, pp_mask = pp_block.final_deriv, pp_block.final_cov, pp_block.inv_cov, pp_block.mask
nn_deriv, nn_cov, nn_inv_cov, nn_mask = nn_block.final_deriv, nn_block.final_cov, nn_block.inv_cov, nn_block.mask
pn_deriv, pn_cov, pn_inv_cov, pn_mask = pn_block.final_deriv, pn_block.final_cov, pn_block.inv_cov, pn_block.mask

# Also update nn variables
for key, block in nn_blocks.items():
    globals()[f"{key}_deriv"] = block.final_deriv
    globals()[f"{key}_cov"] = block.final_cov
    globals()[f"{key}_inv_cov"] = block.inv_cov
    globals()[f"{key}_mask"] = block.mask

# ... repeat for others if downstream code uses them ...

# 2. Combine Blocks
# Kappa All: [pos, neg]
kappa_all_block = FisherDataBlock.combine("kappa_all", [kappa_pos_block, kappa_neg_block], 
                                          kappa_all_deriv, kappa_all_cov, n,
                                          fid_vec=data_slics["kappa_all_mean"],
                                          samples=slics_samples.get("kappa_all"))
kappa_all_inv_cov = kappa_all_block.inv_cov
kappa_all_deriv = kappa_all_block.final_deriv # Update global variable

# NN Scales (nn_1, nn_2, etc.): [pos_same, neg_same, pos_opp, neg_opp]
nn_scale_blocks = {}
for scale in [1, 2, 4, 8]:
    # comps = [nn_blocks[f"nn_pos_same{scale}"], nn_blocks[f"nn_neg_same{scale}"],
    #          nn_blocks[f"nn_pos_opp{scale}"], nn_blocks[f"nn_neg_opp{scale}"]]
    
    key = f"nn_{scale}"
    d_val = globals()[f"{key}_deriv"]
    c_val = globals()[f"{key}_cov"]
    m_val = data_slics[f"{key}_mean"]
    s_val = slics_samples.get(key)
    
    # Re-filter explicitly for combined NN blocks
    b = create_block(key, m_val, d_val, c_val, s_val).filter()
    nn_scale_blocks[scale] = b
    # Assign to global var expected later?
    globals()[f"{key}_inv_cov"] = b.inv_cov
    globals()[f"{key}_deriv"] = b.final_deriv

# NN All: [nn_1, nn_2, nn_4, nn_8]
# nn_all_comps = [nn_scale_blocks[1], nn_scale_blocks[2], nn_scale_blocks[4], nn_scale_blocks[8]]
# Re-filter explicitly for NN All
nn_all_block = create_block("nn_all", data_slics["nn_all_mean"], nn_all_deriv, nn_all_cov, slics_samples.get("nn_all")).filter()
nn_all_inv_cov = nn_all_block.inv_cov
nn_all_deriv = nn_all_block.final_deriv

# All: [kappa_all, nn_all]
# Re-filter explicitly for All
all_block = create_block("all", data_slics["all_mean"], all_deriv, all_cov, slics_samples.get("all")).filter()
all_inv_cov = all_block.inv_cov
all_deriv = all_block.final_deriv

# All and KK: [all, kk]
all_and_kk_block = FisherDataBlock.combine("all_and_kk", [all_block, kk_block], all_and_kk_deriv, all_and_kk_cov, n,
                                           fid_vec=data_slics["all_and_kk_mean"],
                                           samples=slics_samples.get("all_and_kk"))
all_and_kk_inv_cov = all_and_kk_block.inv_cov
all_and_kk_deriv = all_and_kk_block.final_deriv

# All and PDF: [all, kappa] (Wait, check construction: slices_samples['all_and_pdf'] = [all, kappa])
# kappa_block is the one for "kappa" (unsplit)
all_and_pdf_block = FisherDataBlock.combine("all_and_pdf", [all_block, kappa_block], all_and_pdf_deriv, all_and_pdf_cov, n,
                                            fid_vec=data_slics["all_and_pdf_mean"],
                                            samples=slics_samples.get("all_and_pdf"))
all_and_pdf_inv_cov = all_and_pdf_block.inv_cov
all_and_pdf_deriv = all_and_pdf_block.final_deriv

# All Howls and KK: [all, kappa, kk] -> [all_and_pdf, kk] ? 
# Construction: slics_samples['all_howls_and_kk'] = np.hstack([slics_samples['all'], slics_samples['kappa'], slics_samples['kk']])
# Equivalent to [all_and_pdf_block, kk_block]
all_howls_and_kk_block = FisherDataBlock.combine("all_howls_and_kk", [all_and_pdf_block, kk_block], 
                                                 all_howls_and_kk_deriv, all_howls_and_kk_cov, n,
                                                 fid_vec=data_slics["all_howls_and_kk_mean"],
                                                 samples=slics_samples.get("all_howls_and_kk"))
all_howls_and_kk_inv_cov = all_howls_and_kk_block.inv_cov
all_howls_and_kk_deriv = all_howls_and_kk_block.final_deriv

dcorr_block = FisherDataBlock.combine("dcorr", [pp_block, nn_block, pn_block], dcorr_deriv, dcorr_cov, n,
                                      fid_vec=data_slics["dcorr_mean"],
                                      samples=slics_samples.get("dcorr"))
dcorr_inv_cov = dcorr_block.inv_cov
dcorr_deriv = dcorr_block.final_deriv


def build_fisher(name, deriv, inv_cov):
    fisher = deriv @ inv_cov @ deriv.T
    cov = np.linalg.inv(fisher)
    print(f"{name} Fisher:\n{fisher}")
    print(f"{name} 1σ: {np.sqrt(np.diag(cov))}")
    return fisher, cov

blocks = {
    "kappa": (kappa_deriv, kappa_inv_cov),
    "kappa_pos": (kappa_pos_deriv, kappa_pos_inv_cov),
    "kappa_neg": (kappa_neg_deriv, kappa_neg_inv_cov),
    "kappa_all": (kappa_all_deriv, kappa_all_inv_cov),
    "nn_all": (nn_all_deriv, nn_all_inv_cov),
    "all": (all_deriv, all_inv_cov),
    "kk": (kk_deriv, kk_inv_cov),
    "all_and_pdf": (all_and_pdf_deriv, all_and_pdf_inv_cov),
    "all_howls_and_kk": (all_howls_and_kk_deriv, all_howls_and_kk_inv_cov),
    "all_and_kk": (all_and_kk_deriv, all_and_kk_inv_cov),
    "dcorr": (dcorr_deriv, dcorr_inv_cov),
}

fisher_mats = {}
cov_mats = {}
for name, (deriv, inv_cov) in blocks.items():
    fisher_mats[name], cov_mats[name] = build_fisher(name, deriv, inv_cov)


from getdist.gaussian_mixtures import GaussianND
names = ['Om', 'sigma8']
labels = ['\\Omega_{\\rm m}', '\\sigma_{\\rm 8}']
cov_kappa = cov_mats['kappa']
cov_kappa_pos = cov_mats['kappa_pos']
cov_kappa_neg = cov_mats['kappa_neg']
cov_kappa_all = cov_mats['kappa_all']
cov_nn_all = cov_mats['nn_all']
cov_all = cov_mats['all']
cov_kk = cov_mats['kk']
cov_all_and_pdf = cov_mats['all_and_pdf']
cov_all_howls_and_kk = cov_mats['all_howls_and_kk']
cov_all_and_kk = cov_mats['all_and_kk']
cov_dcorr = cov_mats['dcorr']
kappa_samples = GaussianND(mean=params_fid, cov=cov_kappa, names=names, labels=labels, label=r"PDF")
kappa_pos_samples = GaussianND(mean=params_fid, cov=cov_kappa_pos, names=names, labels=labels, label=r"$C^{p}(\kappa)$")
kappa_neg_samples = GaussianND(mean=params_fid, cov=cov_kappa_neg, names=names, labels=labels, label=r"$C^{n}(\kappa)$")
kappa_all_samples = GaussianND(mean=params_fid, cov=cov_kappa_all, names=names, labels=labels, label=r"$C^{\mathrm{all}}(\kappa)$")
nn_all_samples = GaussianND(mean=params_fid, cov=cov_nn_all, names=names, labels=labels, label=r"$C^{\mathrm{all}}_{1, 2, 4, 8}(\theta)$")
all_samples = GaussianND(mean=params_fid, cov=cov_all, names=names, labels=labels, label=r"$DCDP + kNN$")
kk_samples = GaussianND(mean=params_fid, cov=cov_kk, names=names, labels=labels, label=r"$\kappa$-2PCF")
all_and_pdf_samples = GaussianND(mean=params_fid, cov=cov_all_and_pdf, names=names, labels=labels, label=r"$All HOWLS$")
all_howls_and_kk_samples = GaussianND(mean=params_fid, cov=cov_all_howls_and_kk, names=names, labels=labels, label=r"$ALL HOWLS + \kappa\text{-2PCF}$")
all_and_kk_samples = GaussianND(mean=params_fid, cov=cov_all_and_kk, names=names, labels=labels, label=r"$All HOWLS + \kappa\text{-2PCF}$")
dcorr_samples = GaussianND(mean=params_fid, cov=cov_dcorr, names=names, labels=labels, label=r"$D-2PCF$")

import matplotlib.pyplot as plt
from getdist import plots
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 8

# use tab10 colors
color_map = plt.get_cmap('tab10')
# pdf kappa_pos, kappa_neg, kappa_all
g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=0.8)
g.settings.num_plot_contours = 1
g.plot_2d([kk_samples, kappa_samples, kappa_all_samples],
          param1="Om", param2="sigma8",
            filled=False,
            lims = [0.3, 0.325, 0.825, 0.86],
            colors = [color_map(i) for i in range(2)] + ['red'],
            lws=2)
g.add_legend([r"$\kappa$-2PCF", r"PDF", r"D-CDF"], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcdf.png", dpi=300)
# pdf nn_1, nn_2, nn_4, nn_8, nn_all
g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=0.8)
g.settings.num_plot_contours = 1
g.plot_2d([kk_samples, kappa_samples, nn_all_samples],
          param1="Om", param2="sigma8",
            filled=False,
            lims = [0.3, 0.325, 0.825, 0.86],
            colors = [color_map(i) for i in range(2)] + ['red'],
            lws=2)
g.add_legend([r"$\kappa$-2PCF", r"PDF", r"kNNs"], legend_loc='upper right', fontsize=7)
g.export("outfiles/nns.png", dpi=300)

# kappa_all nn_all all
g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=0.8)
g.settings.num_plot_contours = 1
g.plot_2d([kk_samples, kappa_samples, all_samples, all_and_kk_samples, all_and_pdf_samples, all_howls_and_kk_samples],
          param1="Om", param2="sigma8",
            filled=False,
            lims = [0.3, 0.325, 0.825, 0.86],
            colors = [color_map(i) for i in range(4)] + ['red', 'black'],
            ls = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed'],
            lws=2)
g.add_legend([r"$\kappa$-2PCF", r"PDF", r"Defects", r"$\text{Defects} + \kappa\text{-2PCF}$", r"Defects + PDF", r"$\text{Defects + PDF} + \kappa\text{-2PCF}$"], legend_loc='upper right', fontsize=7)
g.export("outfiles/combined.png", dpi=300)
          
g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=0.8)
g.settings.num_plot_contours = 1
g.plot_2d([kk_samples, kappa_samples, dcorr_samples, nn_all_samples],
          param1="Om", param2="sigma8",
            filled=False,
            lims = [0.3, 0.325, 0.825, 0.86],
            colors = [color_map(i) for i in range(3)] + ['red'],
            lws=2)
g.add_legend([r"$\kappa$-2PCF", r"PDF", r"D-2PCF", r"kNNs"], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcorr.png", dpi=300)

# print 1 sigma errors
print("1 sigma errors:")
print(f"PDF kappa: {np.sqrt(np.diag(cov_kappa))}")
print(f"Defect kappa pos: {np.sqrt(np.diag(cov_kappa_pos))}")
print(f"Defect kappa neg: {np.sqrt(np.diag(cov_kappa_neg))}")
print(f"Defect kappa all: {np.sqrt(np.diag(cov_kappa_all))}")
print(f"Defect nn all: {np.sqrt(np.diag(cov_nn_all))}")
print(f"Defect all: {np.sqrt(np.diag(cov_all))}")
print(f"kk: {np.sqrt(np.diag(cov_kk))}")
print(f"All and PDF: {np.sqrt(np.diag(cov_all_and_pdf))}")
print(f"All HOWLS and kk: {np.sqrt(np.diag(cov_all_howls_and_kk))}")
print(f"All and kk: {np.sqrt(np.diag(cov_all_and_kk))}")

# plot errors with horizontal errorbar-like plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define External Data
# Label, rel_err_om (%), rel_err_sigma8 (%)
external_data = [
    (r"$\kappa$-2PCF", 1.88, 1.13),
    (r"$\kappa$-PDF", 0.70, 0.42),
    ("Peaks", 0.70, 0.45),
    ("MFs", 0.74, 0.35),
    ("BNs", 1.27, 0.72),
    ("Pers. BNs", 0.60, 0.32),
    ("Pers. heat.", 0.86, 0.56),
    ("ST", 0.63, 0.39)
]

# Calculate our relative errors in %
fid_om = params_fid[0]
fid_s8 = params_fid[1]

# Labels for our results
our_labels = [r"$\kappa$-2PCF", r"$\kappa$-PDF", "Defects", r"$\text{Defects} + \kappa\text{-2PCF}$", "Defects + PDF", r"Defects + PDF + $\kappa\text{-2PCF}$"]
# Covariances for our results
our_covs = [cov_kk, cov_kappa, cov_all, cov_all_and_kk, cov_all_and_pdf, cov_all_howls_and_kk]

our_data = []
for lbl, cov in zip(our_labels, our_covs):
    err_om = np.sqrt(cov[0,0])
    err_s8 = np.sqrt(cov[1,1])
    rel_om = (err_om / fid_om) * 100
    rel_s8 = (err_s8 / fid_s8) * 100
    our_data.append((lbl, rel_om, rel_s8))

# Reverse the order for plotting top-to-bottom
# We want our results at the top, then external results below
all_data = our_data + external_data

# Create Plot
fig, ax = plt.subplots(figsize=(8, 4))

y_pos = np.arange(len(all_data))

# Extract values
labels = [d[0] for d in all_data]
vals_om = np.array([d[1] for d in all_data])
vals_s8 = np.array([d[2] for d in all_data])

# Calculate intensities (darker = smaller error = stronger constraint)
# We can normalize based on the range of all data displayed
min_om, max_om = vals_om.min(), vals_om.max()
min_s8, max_s8 = vals_s8.min(), vals_s8.max()

# Plot Bars
# Use hlines or plot for thin bars with markers
# Om bars to the left (negative direction for plotting, but labeled positive)
# Sigma8 bars to the right

for i, (om, s8) in enumerate(zip(vals_om, vals_s8)):
    # Calculate intensity/alpha
    # Stronger constraint (smaller value) -> Higher alpha/darker color
    # Let alpha range from 0.4 to 1.0
    alpha_om = 0.8 - 0.6 * (om - min_om) / (max_om - min_om)
    alpha_s8 = 0.8 - 0.6 * (s8 - min_s8) / (max_s8 - min_s8)
    
    # Or map to color map intensity
    # intensity_om = 0.3 + 0.7 * alpha_om (if using cm.Blues(intensity))
    color_om = cm.Blues(0.3 + 0.7 * alpha_om)
    color_s8 = cm.Oranges(0.3 + 0.7 * alpha_s8)
    
    if i == 2:
        ax.plot([0, -om], [i, i], color=color_om, linewidth=2, marker='*', markevery=[1], markersize=10, markerfacecolor=color_om)
        # Plot Sigma8 (Right)
        ax.plot([0, s8], [i, i], color=color_s8, linewidth=2, marker='*', markevery=[1], markersize=10, markerfacecolor=color_s8)
    else:
        # Plot Om (Left)
        ax.plot([0, -om], [i, i], color=color_om, linewidth=2, marker='s', markevery=[1], markersize=6, markerfacecolor=color_om)
        # Plot Sigma8 (Right)
        ax.plot([0, s8], [i, i], color=color_s8, linewidth=2, marker='s', markevery=[1], markersize=6, markerfacecolor=color_s8)

    # Add text labels, times improved over kappa-2PCF (first entry in our data) for our results, and % for all
    # Om Label (left)
    if i == 0:
        ax.text(-om - 0.1, i, f"{om:.2f}%", va='center', ha='right', fontsize=9, color=color_om)
        # Sigma8 Label (right)
        ax.text(s8 + 0.1, i, f"{s8:.2f}%", va='center', ha='left', fontsize=9, color=color_s8)
    else:
        ratio_om = vals_om[0] / om
        ratio_s8 = vals_s8[0] / s8
        ax.text(-om - 0.1, i, f"{ratio_om:.1f}x", va='center', ha='right', fontsize=9, color=color_om)
        # Sigma8 Label (right)
        ax.text(s8 + 0.1, i, f"{ratio_s8:.1f}x", va='center', ha='left', fontsize=9, color=color_s8)

# Dummy plots for legend
ax.plot([], [], color=cm.Blues(0.8), linewidth=2, marker='s', markersize=6, label=r'$\delta \Omega_{\rm m} / \Omega_{\rm m} [\%]$')
ax.plot([], [], color=cm.Oranges(0.8), linewidth=2, marker='s', markersize=6, label=r'$\delta \sigma_8 / \sigma_8 [\%]$')

# text: This Work, Euclid HOWLS
ax.text(-max(vals_om)*1.3 + 0.1, len(our_data)-1, "This Work", va='center', ha='left', fontsize=9)
ax.text(-max(vals_om)*1.3 + 0.1, len(our_data), "Euclid HOWLS", va='center', ha='left', fontsize=9)

# Dashed line separating ours and external
sep_index = len(our_data) - 0.5
ax.axhline(y=sep_index, color='black', linestyle='--', linewidth=1)

# defects results shaded region
ax.axvspan(-vals_om[2], vals_s8[2], color='gray', alpha=0.3)

ax.set_xlim(-max(vals_om)*1.4, max(vals_s8)*1.4)

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis() # Top to bottom

# Set x-axis labels to be positive on both sides
ticks = ax.get_xticks()
ax.set_xticklabels([f"{abs(t):.1f}" for t in ticks])

ax.set_xlabel(r'Relative Precision [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
# legend with text in color, left bottom for Om, right for Sigma8


# Central line
ax.axvline(x=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig("outfiles/1sigma_bars.png", dpi=300)
plt.close()

# save fisher matrix for defects
np.save("outfiles/defects_fisher_s467_th_KL0.1.npy", fisher_mats['all'])
# save pdf results
np.save("outfiles/pdf_fisher_s467_th_KL0.1.npy", fisher_mats['kappa'])
# save kk
np.save("outfiles/kk_fisher_s467_th_KL0.1.npy", fisher_mats['kk'])

# save a npz of all relative errors for 2pcf, pdf, defects, defects+pdf, defects+pdf+2pcf
np.savez("outfiles/relative_errors_s467_th_KL0.1.npz",
         labels=our_labels,
         rel_err_om=[(np.sqrt(cov[0,0]) / fid_om) * 100 for cov in our_covs],
         rel_err_s8=[(np.sqrt(cov[1,1]) / fid_s8) * 100 for cov in our_covs])