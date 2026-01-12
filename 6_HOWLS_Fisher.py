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
data_slics = np.load(f"{SLICS_DIR}/SLICS_stats_cov_mean_s467_subfields.npz")

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
all_and_pdf_deriv = []
all_howls_and_kk_deriv = []
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

area_scaling = 100 / 15000 / 4  # euclid
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

n=924 * 4
kappa_inv_cov = np.linalg.inv(kappa_cov) * (n-28-2) / (n-1)
kappa_pos_inv_cov = np.linalg.inv(kappa_pos_cov) * (n-30-2) / (n-1)
kappa_neg_inv_cov = np.linalg.inv(kappa_neg_cov) * (n-30-2) / (n-1)
nn_pos_same1_inv_cov = np.linalg.inv(nn_pos_same1_cov) * (n-30-2) / (n-1)
nn_neg_same1_inv_cov = np.linalg.inv(nn_neg_same1_cov) * (n-30-2) / (n-1)
nn_pos_opp1_inv_cov = np.linalg.inv(nn_pos_opp1_cov) * (n-30-2) / (n-1)
nn_neg_opp1_inv_cov = np.linalg.inv(nn_neg_opp1_cov) * (n-30-2) / (n-1)
nn_pos_same2_inv_cov = np.linalg.inv(nn_pos_same2_cov) * (n-30-2) / (n-1)
nn_neg_same2_inv_cov = np.linalg.inv(nn_neg_same2_cov) * (n-30-2) / (n-1)  
nn_pos_opp2_inv_cov = np.linalg.inv(nn_pos_opp2_cov) * (n-30-2) / (n-1)
nn_neg_opp2_inv_cov = np.linalg.inv(nn_neg_opp2_cov) * (n-30-2) / (n-1)
nn_pos_same4_inv_cov = np.linalg.inv(nn_pos_same4_cov) * (n-30-2) / (n-1)
nn_neg_same4_inv_cov = np.linalg.inv(nn_neg_same4_cov) * (n-30-2) / (n-1)
nn_pos_opp4_inv_cov = np.linalg.inv(nn_pos_opp4_cov) * (n-30-2) / (n-1)
nn_neg_opp4_inv_cov = np.linalg.inv(nn_neg_opp4_cov) * (n-30-2) / (n-1) 
nn_pos_same8_inv_cov = np.linalg.inv(nn_pos_same8_cov)  * (n-30-2) / (n-1)
nn_neg_same8_inv_cov = np.linalg.inv(nn_neg_same8_cov) * (n-30-2) / (n-1)
nn_pos_opp8_inv_cov = np.linalg.inv(nn_pos_opp8_cov) * (n-30-2) / (n-1)
nn_neg_opp8_inv_cov = np.linalg.inv(nn_neg_opp8_cov)  * (n-30-2) / (n-1)
kappa_all_inv_cov = np.linalg.inv(kappa_all_cov) * (n-30*2-2) / (n-1)
nn_1_inv_cov = np.linalg.inv(nn_1_cov) * (n-30*4-2) / (n-1)
nn_2_inv_cov = np.linalg.inv(nn_2_cov) * (n-30*4-2) / (n-1)
nn_4_inv_cov = np.linalg.inv(nn_4_cov) * (n-30*4-2) / (n-1)
nn_8_inv_cov = np.linalg.inv(nn_8_cov) * (n-30*4-2) / (n-1)
nn_all_inv_cov = np.linalg.inv(nn_all_cov) * (n-30*16-2) / (n-1)
all_inv_cov = np.linalg.inv(all_cov) * (n-30*16-30*2-2) / (n-1)  
kk_inv_cov = np.linalg.inv(kk_cov) * (n-10-2) / (n-1)
all_and_pdf_inv_cov = np.linalg.inv(all_and_pdf_cov) * (n-30*16-30*2-28-2) / (n-1)
all_howls_and_kk_inv_cov = np.linalg.inv(all_howls_and_kk_cov) * (n-30*16-30*2-28-10-2) / (n-1)

fisher_kappa = np.zeros((2,2))
fisher_kappa_pos = np.zeros((2,2))
fisher_kappa_neg = np.zeros((2,2))
fisher_nn_pos_same1 = np.zeros((2,2))
fisher_nn_neg_same1 = np.zeros((2,2))
fisher_nn_pos_opp1 = np.zeros((2,2))
fisher_nn_neg_opp1 = np.zeros((2,2))
fisher_nn_pos_same2 = np.zeros((2,2))   
fisher_nn_neg_same2 = np.zeros((2,2))
fisher_nn_pos_opp2 = np.zeros((2,2))
fisher_nn_neg_opp2 = np.zeros((2,2))
fisher_nn_pos_same4 = np.zeros((2,2))
fisher_nn_neg_same4 = np.zeros((2,2))
fisher_nn_pos_opp4 = np.zeros((2,2))
fisher_nn_neg_opp4 = np.zeros((2,2)) 
fisher_nn_pos_same8 = np.zeros((2,2))
fisher_nn_neg_same8 = np.zeros((2,2))
fisher_nn_pos_opp8 = np.zeros((2,2))
fisher_nn_neg_opp8 = np.zeros((2,2)) 
fisher_kappa_all = np.zeros((2,2))
fisher_nn_1 = np.zeros((2,2))
fisher_nn_2 = np.zeros((2,2))
fisher_nn_4 = np.zeros((2,2))
fisher_nn_8 = np.zeros((2,2))
fisher_nn_all = np.zeros((2,2))
fisher_all = np.zeros((2,2))
fisher_kk = np.zeros((2,2))
fisher_all_and_pdf = np.zeros((2,2))    
fisher_all_howls_and_kk = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        fisher_kappa[i,j] = kappa_deriv[i] @ kappa_inv_cov @ kappa_deriv[j]
        fisher_kappa_pos[i,j] = kappa_pos_deriv[i] @ kappa_pos_inv_cov @ kappa_pos_deriv[j]
        fisher_kappa_neg[i,j] = kappa_neg_deriv[i] @ kappa_neg_inv_cov @ kappa_neg_deriv[j]
        fisher_nn_pos_same1[i,j] = nn_pos_same1_deriv[i] @ nn_pos_same1_inv_cov @ nn_pos_same1_deriv[j]
        fisher_nn_neg_same1[i,j] = nn_neg_same1_deriv[i] @ nn_neg_same1_inv_cov @ nn_neg_same1_deriv[j]
        fisher_nn_pos_opp1[i,j] = nn_pos_opp1_deriv[i] @ nn_pos_opp1_inv_cov @ nn_pos_opp1_deriv[j]
        fisher_nn_neg_opp1[i,j] = nn_neg_opp1_deriv[i] @ nn_neg_opp1_inv_cov @ nn_neg_opp1_deriv[j]
        fisher_nn_pos_same2[i,j] = nn_pos_same2_deriv[i] @ nn_pos_same2_inv_cov @ nn_pos_same2_deriv[j]
        fisher_nn_neg_same2[i,j] = nn_neg_same2_deriv[i] @ nn_neg_same2_inv_cov @ nn_neg_same2_deriv[j]
        fisher_nn_pos_opp2[i,j] = nn_pos_opp2_deriv[i] @ nn_pos_opp2_inv_cov @ nn_pos_opp2_deriv[j]
        fisher_nn_neg_opp2[i,j] = nn_neg_opp2_deriv[i] @ nn_neg_opp2_inv_cov @ nn_neg_opp2_deriv[j]
        fisher_nn_pos_same4[i,j] = nn_pos_same4_deriv[i] @ nn_pos_same4_inv_cov @ nn_pos_same4_deriv[j]
        fisher_nn_neg_same4[i,j] = nn_neg_same4_deriv[i] @ nn_neg_same4_inv_cov @ nn_neg_same4_deriv[j]
        fisher_nn_pos_opp4[i,j] = nn_pos_opp4_deriv[i] @ nn_pos_opp4_inv_cov @ nn_pos_opp4_deriv[j]
        fisher_nn_neg_opp4[i,j] = nn_neg_opp4_deriv[i] @ nn_neg_opp4_inv_cov @ nn_neg_opp4_deriv[j] 
        fisher_nn_pos_same8[i,j] = nn_pos_same8_deriv[i] @ nn_pos_same8_inv_cov @ nn_pos_same8_deriv[j]
        fisher_nn_neg_same8[i,j] = nn_neg_same8_deriv[i] @ nn_neg_same8_inv_cov @ nn_neg_same8_deriv[j]
        fisher_nn_pos_opp8[i,j] = nn_pos_opp8_deriv[i] @ nn_pos_opp8_inv_cov @ nn_pos_opp8_deriv[j]
        fisher_nn_neg_opp8[i,j] = nn_neg_opp8_deriv[i] @ nn_neg_opp8_inv_cov @ nn_neg_opp8_deriv[j] 
        fisher_kappa_all[i,j] = kappa_all_deriv[i] @ kappa_all_inv_cov @ kappa_all_deriv[j]
        fisher_nn_1[i,j] = nn_1_deriv[i] @ nn_1_inv_cov @ nn_1_deriv[j]
        fisher_nn_2[i,j] = nn_2_deriv[i] @ nn_2_inv_cov @ nn_2_deriv[j]
        fisher_nn_4[i,j] = nn_4_deriv[i] @ nn_4_inv_cov @ nn_4_deriv[j]
        fisher_nn_8[i,j] = nn_8_deriv[i] @ nn_8_inv_cov @ nn_8_deriv[j]
        fisher_nn_all[i,j] = nn_all_deriv[i] @ nn_all_inv_cov @ nn_all_deriv[j]
        fisher_all[i,j] = all_deriv[i] @ all_inv_cov @ all_deriv[j]
        fisher_kk[i,j] = kk_deriv[i] @ kk_inv_cov @ kk_deriv[j]
        fisher_all_and_pdf[i,j] = all_and_pdf_deriv[i] @ all_and_pdf_inv_cov @ all_and_pdf_deriv[j]
        fisher_all_howls_and_kk[i,j] = all_howls_and_kk_deriv[i] @ all_howls_and_kk_inv_cov @ all_howls_and_kk_deriv[j]

n = 924
cov_kappa = np.linalg.inv(fisher_kappa) 
print(fisher_kappa, cov_kappa)
cov_kappa_pos = np.linalg.inv(fisher_kappa_pos) 
print(cov_kappa_pos)
cov_kappa_neg = np.linalg.inv(fisher_kappa_neg)
print(np.linalg.det(cov_kappa_neg))
cov_nn_pos_same1 = np.linalg.inv(fisher_nn_pos_same1) 
cov_nn_neg_same1 = np.linalg.inv(fisher_nn_neg_same1) 
cov_nn_pos_opp1 = np.linalg.inv(fisher_nn_pos_opp1) 
cov_nn_neg_opp1 = np.linalg.inv(fisher_nn_neg_opp1) 
cov_nn_pos_same2 = np.linalg.inv(fisher_nn_pos_same2) 
cov_nn_neg_same2 = np.linalg.inv(fisher_nn_neg_same2) 
cov_nn_pos_opp2 = np.linalg.inv(fisher_nn_pos_opp2) 
cov_nn_neg_opp2 = np.linalg.inv(fisher_nn_neg_opp2) 
cov_nn_pos_same4 = np.linalg.inv(fisher_nn_pos_same4) 
cov_nn_neg_same4 = np.linalg.inv(fisher_nn_neg_same4) 
cov_nn_pos_opp4 = np.linalg.inv(fisher_nn_pos_opp4) 
cov_nn_neg_opp4 = np.linalg.inv(fisher_nn_neg_opp4) 
cov_nn_pos_same8 = np.linalg.inv(fisher_nn_pos_same8) 
cov_nn_neg_same8 = np.linalg.inv(fisher_nn_neg_same8) 
cov_nn_pos_opp8 = np.linalg.inv(fisher_nn_pos_opp8) 
cov_nn_neg_opp8 = np.linalg.inv(fisher_nn_neg_opp8) 
cov_kappa_all = np.linalg.inv(fisher_kappa_all) 
print(np.linalg.det(cov_kappa_all))
cov_nn_1 = np.linalg.inv(fisher_nn_1) 
cov_nn_2 = np.linalg.inv(fisher_nn_2) 
cov_nn_4 = np.linalg.inv(fisher_nn_4) 
cov_nn_8 = np.linalg.inv(fisher_nn_8) 
cov_nn_all = np.linalg.inv(fisher_nn_all) 
cov_all = np.linalg.inv(fisher_all) 
cov_kk = np.linalg.inv(fisher_kk)
print(np.linalg.det(cov_kk))
cov_all_and_pdf = np.linalg.inv(fisher_all_and_pdf)
cov_all_howls_and_kk = np.linalg.inv(fisher_all_howls_and_kk)

from getdist.gaussian_mixtures import GaussianND
names = ['Om', 'sigma8']
labels = ['\\Omega_m', '\\sigma_8']

kappa_samples = GaussianND(mean=params_fid, cov=cov_kappa, names=names, labels=labels, label=r"PDF")
kappa_pos_samples = GaussianND(mean=params_fid, cov=cov_kappa_pos, names=names, labels=labels, label=r"$C^{p}(\kappa)$")
kappa_neg_samples = GaussianND(mean=params_fid, cov=cov_kappa_neg, names=names, labels=labels, label=r"$C^{n}(\kappa)$")
kappa_all_samples = GaussianND(mean=params_fid, cov=cov_kappa_all, names=names, labels=labels, label=r"$C^{\mathrm{all}}(\kappa)$")
nn_1_samples = GaussianND(mean=params_fid, cov=cov_nn_1, names=names, labels=labels, label=r"$C^{\mathrm{all}}_1(\theta)$")
nn_2_samples = GaussianND(mean=params_fid, cov=cov_nn_2, names=names, labels=labels, label=r"$C^{\mathrm{all}}_2(\theta)$")
nn_4_samples = GaussianND(mean=params_fid, cov=cov_nn_4, names=names, labels=labels, label=r"$C^{\mathrm{all}}_4(\theta)$")
nn_8_samples = GaussianND(mean=params_fid, cov=cov_nn_8, names=names, labels=labels, label=r"$C^{\mathrm{all}}_8(\theta)$")
nn_all_samples = GaussianND(mean=params_fid, cov=cov_nn_all, names=names, labels=labels, label=r"$C^{\mathrm{all}}_{1, 2, 4, 8}(\theta)$")
all_samples = GaussianND(mean=params_fid, cov=cov_all, names=names, labels=labels, label=r"$DCDP + kNN$")
kk_samples = GaussianND(mean=params_fid, cov=cov_kk, names=names, labels=labels, label=r"$\xi^{\kappa\kappa}(\theta)$")
all_and_pdf_samples = GaussianND(mean=params_fid, cov=cov_all_and_pdf, names=names, labels=labels, label=r"$All HOWLS$")
all_howls_and_kk_samples = GaussianND(mean=params_fid, cov=cov_all_howls_and_kk, names=names, labels=labels, label=r"$ALL HOWLS + \xi^{\kappa\kappa}(\theta)$")
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
            lims = [0.295, 0.33, 0.82, 0.865],
            colors = [color_map(i) for i in range(2)] + ['red'],
            lws=2)
g.add_legend([r"$\xi^{\kappa\kappa}(\theta)$", r"PDF", r"DCDFs"], legend_loc='upper right', fontsize=7)
g.export("outfiles/dcdf.png", dpi=300)
# pdf nn_1, nn_2, nn_4, nn_8, nn_all
g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=0.8)
g.settings.num_plot_contours = 1
g.plot_2d([kk_samples, kappa_samples, nn_all_samples],
          param1="Om", param2="sigma8",
            filled=False,
            lims = [0.295, 0.33, 0.82, 0.865],
            colors = [color_map(i) for i in range(2)] + ['red'],
            lws=2)
g.add_legend([r"$\xi^{\kappa\kappa}(\theta)$", r"PDF", r"kNNs"], legend_loc='upper right', fontsize=7)
g.export("outfiles/nns.png", dpi=300)

# kappa_all nn_all all
g = plots.get_subplot_plotter(width_inch=4, subplot_size_ratio=0.8)
g.settings.num_plot_contours = 1
g.plot_2d([kk_samples, kappa_samples, all_samples, all_and_pdf_samples, all_howls_and_kk_samples],
          param1="Om", param2="sigma8",
            filled=False,
            lims = [0.295, 0.33, 0.82, 0.865],
            colors = [color_map(i) for i in range(3)] + ['red', 'black'],
            ls = ['solid', 'solid', 'solid', 'solid', 'dashed'],
            lws=2)
g.add_legend([r"$\xi^{\kappa\kappa}(\theta)$", r"PDF", r"Defects", r"All HOWLS", r"$\text{All HOWLS} + \xi^{\kappa\kappa}(\theta)$"], legend_loc='upper right', fontsize=7)
g.export("outfiles/combined.png", dpi=300)
          


# print 1 sigma errors
print("1 sigma errors:")
print(f"PDF kappa: {np.sqrt(np.diag(cov_kappa))}")
print(f"Defect kappa pos: {np.sqrt(np.diag(cov_kappa_pos))}")
print(f"Defect kappa neg: {np.sqrt(np.diag(cov_kappa_neg))}")
print(f"Defect kappa all: {np.sqrt(np.diag(cov_kappa_all))}")
print(f"Defect nn 1: {np.sqrt(np.diag(cov_nn_1))}")
print(f"Defect nn 2: {np.sqrt(np.diag(cov_nn_2))}")
print(f"Defect nn 4: {np.sqrt(np.diag(cov_nn_4))}")
print(f"Defect nn 8: {np.sqrt(np.diag(cov_nn_8))}")
print(f"Defect nn all: {np.sqrt(np.diag(cov_nn_all))}")
print(f"Defect all: {np.sqrt(np.diag(cov_all))}")
print(f"kk: {np.sqrt(np.diag(cov_kk))}")
print(f"All and PDF: {np.sqrt(np.diag(cov_all_and_pdf))}")
print(f"All HOWLS and kk: {np.sqrt(np.diag(cov_all_howls_and_kk))}")

# plot errors with horizontal errorbar-like plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
labels = [r"$\xi^{\kappa\kappa}(\theta)$", "PDF", "Defects", "All HOWLS", r"All HOWLS + $\xi^{\kappa\kappa}(\theta)$"]
sigma_om = np.array([np.sqrt(cov_kk[0,0]), np.sqrt(cov_kappa[0,0]), np.sqrt(cov_all[0,0]), np.sqrt(cov_all_and_pdf[0,0]), np.sqrt(cov_all_howls_and_kk[0,0])])
sigma_sigma8 = np.array([np.sqrt(cov_kk[1,1]), np.sqrt(cov_kappa[1,1]), np.sqrt(cov_all[1,1]), np.sqrt(cov_all_and_pdf[1,1]), np.sqrt(cov_all_howls_and_kk[1,1])])

# Normalize for color intensity (smaller error = darker color, stronger constraint)
norm_om = 0.9 - (sigma_om - sigma_om.min()) / (sigma_om.max() - sigma_om.min())
norm_sigma8 = 0.9 - (sigma_sigma8 - sigma_sigma8.min()) / (sigma_sigma8.max() - sigma_sigma8.min())
max_err = max(sigma_om.max(), sigma_sigma8.max())

x = np.arange(len(labels))
width = 0.35  # the offset between the two error bars

fig, ax = plt.subplots(figsize=(5, 3))

# Plot Om_m errors
for i, (y_pos, err, intensity) in enumerate(zip(x - width/2, sigma_om, norm_om)):
    color = cm.Blues(0.3 + 0.7 * intensity)  # Darker = stronger constraint
    ax.plot([0, err], [y_pos, y_pos], color=color, linewidth=3, marker='s', markevery=[1], markersize=6, markerfacecolor=color)
    if i == 0:  # label baseline xi row
        ax.text(err + 0.02 * max_err, y_pos, fr"{err:.4f}", va='center', ha='left', color=color, fontsize=7)
    if i > 0:  # annotate percent reduction vs xi
        reduction = (sigma_om[0] - err) / sigma_om[0] * 100
        ax.text(err + 0.02 * max_err, y_pos, f"{reduction:.0f}%", va='center', ha='left', color=color, fontsize=7)

# Plot sigma_8 errors  
for i, (y_pos, err, intensity) in enumerate(zip(x + width/2, sigma_sigma8, norm_sigma8)):
    color = cm.Oranges(0.3 + 0.7 * intensity)  # Darker = stronger constraint
    ax.plot([0, err], [y_pos, y_pos], color=color, linewidth=3, marker='s', markevery=[1], markersize=6, markerfacecolor=color)
    if i == 0:  # label baseline xi row
        ax.text(err + 0.02 * max_err, y_pos, fr"{err:.4f}", va='center', ha='left', color=color, fontsize=7)
    if i > 0:  # annotate percent reduction vs xi
        reduction = (sigma_sigma8[0] - err) / sigma_sigma8[0] * 100
        ax.text(err + 0.02 * max_err, y_pos, f"{reduction:.0f}%", va='center', ha='left', color=color, fontsize=7)

# Add dummy plots for legend
ax.plot([], [], color=cm.Blues(0.8), linewidth=3, marker='s', markersize=6, label=r'$\Delta {\Omega_m}$')
ax.plot([], [], color=cm.Oranges(0.8), linewidth=3, marker='s', markersize=6, label=r'$\Delta {\sigma_8}$')
ax.set_xlabel(r'1 $\sigma$ Range')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.tick_params(axis='y', which='both', length=0)  # Remove all y-axis tick marks
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlim(0, max_err * 1.15)
ax.legend()
plt.tight_layout()
plt.savefig("outfiles/1sigma_bars.png", dpi=300)
plt.close()