from DefectFinder import DefectFinder
from tqdm import tqdm
import numpy as np
import os

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

N = 128
for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.02, 0.03, 30)
    df = DefectFinder(f"{OM02_DIR}/angles_s467_{i}_LCDM_Om02_ks_nomask_shear.npz")
    df.load_defects(f"{OM02_DIR}/defects_s467_{i}_LCDM_Om02_ks_nomask_shear.npz")
    kappa, _ = np.histogram(df.kappa, bins=bin_edges, density=True)
    kappa_pos = df.convergence_dist(charge=0.5)
    kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bin_edges]) / 25 * (180/np.pi)**2 
    kappa_neg = df.convergence_dist(charge=-0.5)
    kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bin_edges]) / 25 * (180/np.pi)**2

    nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
    nn_neg_same1 = df.nearest_distance(charge=-0.5, k=1) 
    nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
    nn_neg_opp1 = df.nearest_opposite_distance(charge=-0.5, k=1)
    nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
    nn_neg_same2 = df.nearest_distance(charge=-0.5, k=2)
    nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
    nn_neg_opp2 = df.nearest_opposite_distance(charge=-0.5, k=2)
    nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
    nn_neg_same4 = df.nearest_distance(charge=-0.5, k=4)
    nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
    nn_neg_opp4 = df.nearest_opposite_distance(charge=-0.5, k=4)
    nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
    nn_neg_same8 = df.nearest_distance(charge=-0.5, k=8)
    nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
    nn_neg_opp8 = df.nearest_opposite_distance(charge=-0.5, k=8)
    bin_edges1 = np.linspace(5, 50, 30)
    bin_edges2 = np.linspace(10, 60, 30)
    bin_edges4 = np.linspace(20, 60, 30)
    bin_edges8 = np.linspace(30, 100, 30)
    nn_pos_same1_cum = np.array([np.sum(nn_pos_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_neg_same1_cum = np.array([np.sum(nn_neg_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_opp1_cum = np.array([np.sum(nn_pos_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_neg_opp1_cum = np.array([np.sum(nn_neg_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_same2_cum = np.array([np.sum(nn_pos_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_same2_cum = np.array([np.sum(nn_neg_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_opp2_cum = np.array([np.sum(nn_pos_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_opp2_cum = np.array([np.sum(nn_neg_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_same4_cum = np.array([np.sum(nn_pos_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_same4_cum = np.array([np.sum(nn_neg_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_opp4_cum = np.array([np.sum(nn_pos_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_opp4_cum = np.array([np.sum(nn_neg_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_same8_cum = np.array([np.sum(nn_pos_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_same8_cum = np.array([np.sum(nn_neg_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2 
    nn_pos_opp8_cum = np.array([np.sum(nn_pos_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_opp8_cum = np.array([np.sum(nn_neg_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2

    df = DefectFinder(f"{OM02_DIR}/angles_{i}_LCDM_Om02_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.8, max_sep=10, n_bins=10)

    np.savez(f"{OM02_DIR}/OM02_stats_s467_{i}.npz",
                kappa=kappa,
                kappa_pos_cum=kappa_pos_cum,
                kappa_neg_cum=kappa_neg_cum,
                nn_pos_same1_cum=nn_pos_same1_cum,
                nn_neg_same1_cum=nn_neg_same1_cum,
                nn_pos_opp1_cum=nn_pos_opp1_cum,
                nn_neg_opp1_cum=nn_neg_opp1_cum,
                nn_pos_same2_cum=nn_pos_same2_cum,
                nn_neg_same2_cum=nn_neg_same2_cum,
                nn_pos_opp2_cum=nn_pos_opp2_cum,
                nn_neg_opp2_cum=nn_neg_opp2_cum,
                nn_pos_same4_cum=nn_pos_same4_cum,
                nn_neg_same4_cum=nn_neg_same4_cum,
                nn_pos_opp4_cum=nn_pos_opp4_cum,
                nn_neg_opp4_cum=nn_neg_opp4_cum,
                nn_pos_same8_cum=nn_pos_same8_cum,
                nn_neg_same8_cum=nn_neg_same8_cum,
                nn_pos_opp8_cum=nn_pos_opp8_cum,
                nn_neg_opp8_cum=nn_neg_opp8_cum,
                kk=kk,
                theta=theta,
                bin_edges=bin_edges,
                bin_edges1=bin_edges1,
                bin_edges2=bin_edges2,
                bin_edges4=bin_edges4,
                bin_edges8=bin_edges8)

    
for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.02, 0.03, 30)
    df = DefectFinder(f"{OM04_DIR}/angles_s467_{i}_LCDM_Om04_ks_nomask_shear.npz")
    df.load_defects(f"{OM04_DIR}/defects_s467_{i}_LCDM_Om04_ks_nomask_shear.npz")
    kappa, _ = np.histogram(df.kappa, bins=bin_edges, density=True)
    kappa_pos = df.convergence_dist(charge=0.5)
    kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bin_edges]) / 25 * (180/np.pi)**2 
    kappa_neg = df.convergence_dist(charge=-0.5)
    kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bin_edges]) / 25 * (180/np.pi)**2

    nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
    nn_neg_same1 = df.nearest_distance(charge=-0.5, k=1) 
    nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
    nn_neg_opp1 = df.nearest_opposite_distance(charge=-0.5, k=1)
    nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
    nn_neg_same2 = df.nearest_distance(charge=-0.5, k=2)
    nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
    nn_neg_opp2 = df.nearest_opposite_distance(charge=-0.5, k=2)
    nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
    nn_neg_same4 = df.nearest_distance(charge=-0.5, k=4)
    nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
    nn_neg_opp4 = df.nearest_opposite_distance(charge=-0.5, k=4)
    nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
    nn_neg_same8 = df.nearest_distance(charge=-0.5, k=8)
    nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
    nn_neg_opp8 = df.nearest_opposite_distance(charge=-0.5, k=8)
    bin_edges1 = np.linspace(5, 50, 30)
    bin_edges2 = np.linspace(10, 60, 30)
    bin_edges4 = np.linspace(20, 60, 30)
    bin_edges8 = np.linspace(30, 100, 30)
    nn_pos_same1_cum = np.array([np.sum(nn_pos_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_neg_same1_cum = np.array([np.sum(nn_neg_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_opp1_cum = np.array([np.sum(nn_pos_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_neg_opp1_cum = np.array([np.sum(nn_neg_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_same2_cum = np.array([np.sum(nn_pos_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_same2_cum = np.array([np.sum(nn_neg_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_opp2_cum = np.array([np.sum(nn_pos_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_opp2_cum = np.array([np.sum(nn_neg_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_same4_cum = np.array([np.sum(nn_pos_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_same4_cum = np.array([np.sum(nn_neg_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_opp4_cum = np.array([np.sum(nn_pos_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_opp4_cum = np.array([np.sum(nn_neg_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_same8_cum = np.array([np.sum(nn_pos_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_same8_cum = np.array([np.sum(nn_neg_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2 
    nn_pos_opp8_cum = np.array([np.sum(nn_pos_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_opp8_cum = np.array([np.sum(nn_neg_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2

    df = DefectFinder(f"{OM04_DIR}/angles_{i}_LCDM_Om04_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.8, max_sep=10, n_bins=10)

    np.savez(f"{OM04_DIR}/OM04_stats_s467_{i}.npz",
                kappa=kappa,
                kappa_pos_cum=kappa_pos_cum,
                kappa_neg_cum=kappa_neg_cum,
                nn_pos_same1_cum=nn_pos_same1_cum,
                nn_neg_same1_cum=nn_neg_same1_cum,
                nn_pos_opp1_cum=nn_pos_opp1_cum,
                nn_neg_opp1_cum=nn_neg_opp1_cum,
                nn_pos_same2_cum=nn_pos_same2_cum,
                nn_neg_same2_cum=nn_neg_same2_cum,
                nn_pos_opp2_cum=nn_pos_opp2_cum,
                nn_neg_opp2_cum=nn_neg_opp2_cum,
                nn_pos_same4_cum=nn_pos_same4_cum,
                nn_neg_same4_cum=nn_neg_same4_cum,
                nn_pos_opp4_cum=nn_pos_opp4_cum,
                nn_neg_opp4_cum=nn_neg_opp4_cum,
                nn_pos_same8_cum=nn_pos_same8_cum,
                nn_neg_same8_cum=nn_neg_same8_cum,
                nn_pos_opp8_cum=nn_pos_opp8_cum,
                nn_neg_opp8_cum=nn_neg_opp8_cum,
                kk=kk,
                theta=theta,
                bin_edges=bin_edges,
                bin_edges1=bin_edges1,
                bin_edges2=bin_edges2,
                bin_edges4=bin_edges4,
                bin_edges8=bin_edges8)

for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.02, 0.03, 30)
    df = DefectFinder(f"{S807_DIR}/angles_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz")
    df.load_defects(f"{S807_DIR}/defects_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz")
    kappa, _ = np.histogram(df.kappa, bins=bin_edges, density=True)
    kappa_pos = df.convergence_dist(charge=0.5)
    kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bin_edges]) / 25 * (180/np.pi)**2 
    kappa_neg = df.convergence_dist(charge=-0.5)
    kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bin_edges]) / 25 * (180/np.pi)**2

    nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
    nn_neg_same1 = df.nearest_distance(charge=-0.5, k=1) 
    nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
    nn_neg_opp1 = df.nearest_opposite_distance(charge=-0.5, k=1)
    nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
    nn_neg_same2 = df.nearest_distance(charge=-0.5, k=2)
    nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
    nn_neg_opp2 = df.nearest_opposite_distance(charge=-0.5, k=2)
    nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
    nn_neg_same4 = df.nearest_distance(charge=-0.5, k=4)
    nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
    nn_neg_opp4 = df.nearest_opposite_distance(charge=-0.5, k=4)
    nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
    nn_neg_same8 = df.nearest_distance(charge=-0.5, k=8)
    nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
    nn_neg_opp8 = df.nearest_opposite_distance(charge=-0.5, k=8)
    bin_edges1 = np.linspace(5, 50, 30)
    bin_edges2 = np.linspace(10, 60, 30)
    bin_edges4 = np.linspace(20, 60, 30)
    bin_edges8 = np.linspace(30, 100, 30)
    nn_pos_same1_cum = np.array([np.sum(nn_pos_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2
    nn_neg_same1_cum = np.array([np.sum(nn_neg_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_opp1_cum = np.array([np.sum(nn_pos_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_neg_opp1_cum = np.array([np.sum(nn_neg_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_same2_cum = np.array([np.sum(nn_pos_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_same2_cum = np.array([np.sum(nn_neg_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_opp2_cum = np.array([np.sum(nn_pos_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_opp2_cum = np.array([np.sum(nn_neg_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_same4_cum = np.array([np.sum(nn_pos_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_same4_cum = np.array([np.sum(nn_neg_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_opp4_cum = np.array([np.sum(nn_pos_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_opp4_cum = np.array([np.sum(nn_neg_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_same8_cum = np.array([np.sum(nn_pos_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_same8_cum = np.array([np.sum(nn_neg_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2 
    nn_pos_opp8_cum = np.array([np.sum(nn_pos_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_opp8_cum = np.array([np.sum(nn_neg_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2 

    df = DefectFinder(f"{S807_DIR}/angles_{i}_LCDM_s80707210_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.8, max_sep=10, n_bins=10)

    np.savez(f"{S807_DIR}/S807_stats_s467_{i}.npz",
                kappa=kappa,
                kappa_pos_cum=kappa_pos_cum,
                kappa_neg_cum=kappa_neg_cum,
                nn_pos_same1_cum=nn_pos_same1_cum,
                nn_neg_same1_cum=nn_neg_same1_cum,
                nn_pos_opp1_cum=nn_pos_opp1_cum,
                nn_neg_opp1_cum=nn_neg_opp1_cum,
                nn_pos_same2_cum=nn_pos_same2_cum,
                nn_neg_same2_cum=nn_neg_same2_cum,
                nn_pos_opp2_cum=nn_pos_opp2_cum,
                nn_neg_opp2_cum=nn_neg_opp2_cum,
                nn_pos_same4_cum=nn_pos_same4_cum,
                nn_neg_same4_cum=nn_neg_same4_cum,
                nn_pos_opp4_cum=nn_pos_opp4_cum,
                nn_neg_opp4_cum=nn_neg_opp4_cum,
                nn_pos_same8_cum=nn_pos_same8_cum,
                nn_neg_same8_cum=nn_neg_same8_cum,
                nn_pos_opp8_cum=nn_pos_opp8_cum,
                nn_neg_opp8_cum=nn_neg_opp8_cum,
                kk=kk,
                theta=theta,
                bin_edges=bin_edges,
                bin_edges1=bin_edges1,
                bin_edges2=bin_edges2,
                bin_edges4=bin_edges4,
                bin_edges8=bin_edges8)

for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.02, 0.03, 30)
    df = DefectFinder(f"{S809_DIR}/angles_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz")
    df.load_defects(f"{S809_DIR}/defects_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz")
    kappa, _ = np.histogram(df.kappa, bins=bin_edges, density=True)
    kappa_pos = df.convergence_dist(charge=0.5)
    kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bin_edges]) / 25 * (180/np.pi)**2 
    kappa_neg = df.convergence_dist(charge=-0.5)
    kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bin_edges]) / 25 * (180/np.pi)**2

    nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
    nn_neg_same1 = df.nearest_distance(charge=-0.5, k=1) 
    nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
    nn_neg_opp1 = df.nearest_opposite_distance(charge=-0.5, k=1)
    nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
    nn_neg_same2 = df.nearest_distance(charge=-0.5, k=2)
    nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
    nn_neg_opp2 = df.nearest_opposite_distance(charge=-0.5, k=2)
    nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
    nn_neg_same4 = df.nearest_distance(charge=-0.5, k=4)
    nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
    nn_neg_opp4 = df.nearest_opposite_distance(charge=-0.5, k=4)
    nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
    nn_neg_same8 = df.nearest_distance(charge=-0.5, k=8)
    nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
    nn_neg_opp8 = df.nearest_opposite_distance(charge=-0.5, k=8)
    bin_edges1 = np.linspace(5, 50, 30)
    bin_edges2 = np.linspace(10, 60, 30)
    bin_edges4 = np.linspace(20, 60, 30)
    bin_edges8 = np.linspace(30, 100, 30)
    nn_pos_same1_cum = np.array([np.sum(nn_pos_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2   
    nn_neg_same1_cum = np.array([np.sum(nn_neg_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_opp1_cum = np.array([np.sum(nn_pos_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_neg_opp1_cum = np.array([np.sum(nn_neg_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_same2_cum = np.array([np.sum(nn_pos_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_same2_cum = np.array([np.sum(nn_neg_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_opp2_cum = np.array([np.sum(nn_pos_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_neg_opp2_cum = np.array([np.sum(nn_neg_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_same4_cum = np.array([np.sum(nn_pos_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_same4_cum = np.array([np.sum(nn_neg_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_opp4_cum = np.array([np.sum(nn_pos_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_neg_opp4_cum = np.array([np.sum(nn_neg_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_same8_cum = np.array([np.sum(nn_pos_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_same8_cum = np.array([np.sum(nn_neg_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2 
    nn_pos_opp8_cum = np.array([np.sum(nn_pos_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_neg_opp8_cum = np.array([np.sum(nn_neg_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2 

    df = DefectFinder(f"{S809_DIR}/angles_{i}_LCDM_s80976624_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.8, max_sep=10, n_bins=10)

    np.savez(f"{S809_DIR}/S809_stats_s467_{i}.npz",
                kappa=kappa,
                kappa_pos_cum=kappa_pos_cum,
                kappa_neg_cum=kappa_neg_cum,
                nn_pos_same1_cum=nn_pos_same1_cum,
                nn_neg_same1_cum=nn_neg_same1_cum,
                nn_pos_opp1_cum=nn_pos_opp1_cum,
                nn_neg_opp1_cum=nn_neg_opp1_cum,
                nn_pos_same2_cum=nn_pos_same2_cum,
                nn_neg_same2_cum=nn_neg_same2_cum,
                nn_pos_opp2_cum=nn_pos_opp2_cum,
                nn_neg_opp2_cum=nn_neg_opp2_cum,
                nn_pos_same4_cum=nn_pos_same4_cum,
                nn_neg_same4_cum=nn_neg_same4_cum,
                nn_pos_opp4_cum=nn_pos_opp4_cum,
                nn_neg_opp4_cum=nn_neg_opp4_cum,
                nn_pos_same8_cum=nn_pos_same8_cum,
                nn_neg_same8_cum=nn_neg_same8_cum,
                nn_pos_opp8_cum=nn_pos_opp8_cum,
                nn_neg_opp8_cum=nn_neg_opp8_cum,
                kk=kk,
                theta=theta,
                bin_edges=bin_edges,
                bin_edges1=bin_edges1,
                bin_edges2=bin_edges2,
                bin_edges4=bin_edges4,
                bin_edges8=bin_edges8)

N = 1000
for i in tqdm(range(N)):
    try:
        bin_edges = np.linspace(-0.02, 0.03, 30)
        df = DefectFinder(f"{SLICS_DIR}/angles_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        df.load_defects(f"{SLICS_DIR}/defects_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        kappa, _ = np.histogram(df.kappa, bins=bin_edges, density=True)
        kappa_pos = df.convergence_dist(charge=0.5)
        kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bin_edges]) / 100 * (180/np.pi)**2 
        kappa_neg = df.convergence_dist(charge=-0.5)
        kappa_neg_cum = np.array([np.sum(kappa_neg < b) for b in bin_edges]) / 100 * (180/np.pi)**2

        nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
        nn_neg_same1 = df.nearest_distance(charge=-0.5, k=1) 
        nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
        nn_neg_opp1 = df.nearest_opposite_distance(charge=-0.5, k=1)
        nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
        nn_neg_same2 = df.nearest_distance(charge=-0.5, k=2)
        nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
        nn_neg_opp2 = df.nearest_opposite_distance(charge=-0.5, k=2)
        nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
        nn_neg_same4 = df.nearest_distance(charge=-0.5, k=4)
        nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
        nn_neg_opp4 = df.nearest_opposite_distance(charge=-0.5, k=4)
        nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
        nn_neg_same8 = df.nearest_distance(charge=-0.5, k=8)
        nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
        nn_neg_opp8 = df.nearest_opposite_distance(charge=-0.5, k=8)
        bin_edges1 = np.linspace(5, 50, 30)
        bin_edges2 = np.linspace(10, 60, 30)
        bin_edges4 = np.linspace(20, 60, 30)
        bin_edges8 = np.linspace(30, 100, 30)
        nn_pos_same1_cum = np.array([np.sum(nn_pos_same1 < b) for b in bin_edges1]) / 100 * (180/np.pi)**2   
        nn_neg_same1_cum = np.array([np.sum(nn_neg_same1 < b) for b in bin_edges1]) / 100 * (180/np.pi)**2 
        nn_pos_opp1_cum = np.array([np.sum(nn_pos_opp1 < b) for b in bin_edges1]) / 100 * (180/np.pi)**2 
        nn_neg_opp1_cum = np.array([np.sum(nn_neg_opp1 < b) for b in bin_edges1]) / 100 * (180/np.pi)**2 
        nn_pos_same2_cum = np.array([np.sum(nn_pos_same2 < b) for b in bin_edges2]) / 100 * (180/np.pi)**2 
        nn_neg_same2_cum = np.array([np.sum(nn_neg_same2 < b) for b in bin_edges2]) / 100 * (180/np.pi)**2 
        nn_pos_opp2_cum = np.array([np.sum(nn_pos_opp2 < b) for b in bin_edges2]) / 100 * (180/np.pi)**2 
        nn_neg_opp2_cum = np.array([np.sum(nn_neg_opp2 < b) for b in bin_edges2]) / 100 * (180/np.pi)**2 
        nn_pos_same4_cum = np.array([np.sum(nn_pos_same4 < b) for b in bin_edges4]) / 100 * (180/np.pi)**2 
        nn_neg_same4_cum = np.array([np.sum(nn_neg_same4 < b) for b in bin_edges4]) / 100 * (180/np.pi)**2 
        nn_pos_opp4_cum = np.array([np.sum(nn_pos_opp4 < b) for b in bin_edges4]) / 100 * (180/np.pi)**2 
        nn_neg_opp4_cum = np.array([np.sum(nn_neg_opp4 < b) for b in bin_edges4]) / 100 * (180/np.pi)**2 
        nn_pos_same8_cum = np.array([np.sum(nn_pos_same8 < b) for b in bin_edges8]) / 100 * (180/np.pi)**2
        nn_neg_same8_cum = np.array([np.sum(nn_neg_same8 < b) for b in bin_edges8]) / 100 * (180/np.pi)**2 
        nn_pos_opp8_cum = np.array([np.sum(nn_pos_opp8 < b) for b in bin_edges8]) / 100 * (180/np.pi)**2
        nn_neg_opp8_cum = np.array([np.sum(nn_neg_opp8 < b) for b in bin_edges8]) / 100 * (180/np.pi)**2 

        df = DefectFinder(f"{SLICS_DIR}/angles_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        theta, kk = df.convergence_correlation(min_sep=0.8, max_sep=10, n_bins=10)

        np.savez(f"{SLICS_DIR}/SLICS_stats_s467_{i}.npz",
                    kappa=kappa,
                    kappa_pos_cum=kappa_pos_cum,
                    kappa_neg_cum=kappa_neg_cum,
                    nn_pos_same1_cum=nn_pos_same1_cum,
                    nn_neg_same1_cum=nn_neg_same1_cum,
                    nn_pos_opp1_cum=nn_pos_opp1_cum,
                    nn_neg_opp1_cum=nn_neg_opp1_cum,
                    nn_pos_same2_cum=nn_pos_same2_cum,
                    nn_neg_same2_cum=nn_neg_same2_cum,
                    nn_pos_opp2_cum=nn_pos_opp2_cum,
                    nn_neg_opp2_cum=nn_neg_opp2_cum,
                    nn_pos_same4_cum=nn_pos_same4_cum,
                    nn_neg_same4_cum=nn_neg_same4_cum,
                    nn_pos_opp4_cum=nn_pos_opp4_cum,    
                    nn_neg_opp4_cum=nn_neg_opp4_cum,
                    nn_pos_same8_cum=nn_pos_same8_cum,
                    nn_neg_same8_cum=nn_neg_same8_cum,
                    nn_pos_opp8_cum=nn_pos_opp8_cum,
                    nn_neg_opp8_cum=nn_neg_opp8_cum,
                    kk=kk,
                    theta=theta,
                    bin_edges=bin_edges,
                    bin_edges1=bin_edges1,
                    bin_edges2=bin_edges2,
                    bin_edges4=bin_edges4,
                    bin_edges8=bin_edges8)
    except FileNotFoundError:
        print(f"File for realisation {i} not found, skipping.")
