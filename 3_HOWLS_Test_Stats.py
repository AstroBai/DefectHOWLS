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
kappa_list = []
kappa_dist_list = []
nn_pos_same1_list = []
nn_pos_opp1_list = []
nn_pos_same2_list = []
nn_pos_opp2_list = []
nn_pos_same4_list = []
nn_pos_opp4_list = []
nn_pos_same8_list = []
nn_pos_opp8_list = []
kk_list = []
for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.02, 0.03, 30)
    df = DefectFinder(f"{OM02_DIR}/angles_s467_{i}_LCDM_Om02_ks_nomask_shear.npz")
    df.load_defects(f"{OM02_DIR}/defects_s467_{i}_LCDM_Om02_ks_nomask_shear.npz")
    kappa, _ = np.histogram(df.kappa, bins=bin_edges, density=True)
    kappa_pos = df.convergence_dist(charge=0.5)
    kappa_pos_cum = np.array([np.sum(kappa_pos < b) for b in bin_edges]) / 25 * (180/np.pi)**2 
    kappa_dist_list.append(kappa_pos_cum)

    nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
    nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
    nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
    nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
    nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
    nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
    nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
    nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
    bin_edges1 = np.linspace(0, 100, 30)
    bin_edges2 = np.linspace(0, 100, 30)
    bin_edges4 = np.linspace(0, 100, 30)
    bin_edges8 = np.linspace(0, 100, 30)
    nn_pos_same1_cum = np.array([np.sum(nn_pos_same1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_opp1_cum = np.array([np.sum(nn_pos_opp1 < b) for b in bin_edges1]) / 25 * (180/np.pi)**2 
    nn_pos_same2_cum = np.array([np.sum(nn_pos_same2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_opp2_cum = np.array([np.sum(nn_pos_opp2 < b) for b in bin_edges2]) / 25 * (180/np.pi)**2 
    nn_pos_same4_cum = np.array([np.sum(nn_pos_same4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_opp4_cum = np.array([np.sum(nn_pos_opp4 < b) for b in bin_edges4]) / 25 * (180/np.pi)**2 
    nn_pos_same8_cum = np.array([np.sum(nn_pos_same8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2 
    nn_pos_opp8_cum = np.array([np.sum(nn_pos_opp8 < b) for b in bin_edges8]) / 25 * (180/np.pi)**2
    nn_pos_same1_list.append(nn_pos_same1_cum)
    nn_pos_opp1_list.append(nn_pos_opp1_cum)
    nn_pos_same2_list.append(nn_pos_same2_cum)
    nn_pos_opp2_list.append(nn_pos_opp2_cum)
    nn_pos_same4_list.append(nn_pos_same4_cum)
    nn_pos_opp4_list.append(nn_pos_opp4_cum)
    nn_pos_same8_list.append(nn_pos_same8_cum)
    nn_pos_opp8_list.append(nn_pos_opp8_cum)
    kappa_list.append(kappa)

    df = DefectFinder(f"{OM02_DIR}/angles_{i}_LCDM_Om02_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.8, max_sep=10, n_bins=10)
    kk_list.append(kk)
    

kappa_list = np.array(kappa_list)
kappa_dist_list = np.array(kappa_dist_list)
nn_pos_same1_list = np.array(nn_pos_same1_list)
nn_pos_opp1_list = np.array(nn_pos_opp1_list)
nn_pos_same2_list = np.array(nn_pos_same2_list)
nn_pos_opp2_list = np.array(nn_pos_opp2_list)
nn_pos_same4_list = np.array(nn_pos_same4_list)
nn_pos_opp4_list = np.array(nn_pos_opp4_list)
nn_pos_same8_list = np.array(nn_pos_same8_list)
nn_pos_opp8_list = np.array(nn_pos_opp8_list)
kk_list = np.array(kk_list)

import matplotlib.pyplot as plt

plt.figure()
for i in range(N):
    plt.plot(bin_edges, kappa_dist_list[i])

plt.xlabel("Kappa")
plt.ylabel("Cumulative Distribution")
plt.title("Cumulative Distribution of Kappa for OM02 Realisations")
plt.savefig("outfiles/kappa_cumulative_distribution_OM02.png")

plt.figure()
for i in range(N):
    plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), kappa_list[i])
plt.xlabel("Kappa Bins")
plt.ylabel("Density")
plt.title("Kappa Histogram for OM02 Realisations")
plt.savefig("outfiles/kappa_histogram_OM02.png")

plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
for i in range(N):
    plt.plot(bin_edges1, nn_pos_same1_list[i])
    # plt.plot(bin_edges1, nn_pos_opp1_list[i])
plt.xlabel("Distance")
plt.ylabel("Numbers")
plt.title("Cumulative Distribution of Nearest Neighbor Distances (k=1) for OM02 Realisations")

plt.subplot(1, 4, 2)
for i in range(N):
    plt.plot(bin_edges2, nn_pos_same2_list[i])
    # plt.plot(bin_edges2, nn_pos_opp2_list[i])
plt.xlabel("Distance")
plt.ylabel("Numbers")
plt.title("Cumulative Distribution of Nearest Neighbor Distances (k=2) for OM02 Realisations")

plt.subplot(1, 4, 3)
for i in range(N):
    plt.plot(bin_edges4, nn_pos_same4_list[i])
    # plt.plot(bin_edges4, nn_pos_opp4_list[i])
plt.xlabel("Distance")
plt.ylabel("Numbers")
plt.title("Cumulative Distribution of Nearest Neighbor Distances (k=4) for OM02 Realisations")

plt.subplot(1, 4, 4)
for i in range(N):
    plt.plot(bin_edges8, nn_pos_same8_list[i])
    # plt.plot(bin_edges8, nn_pos_opp8_list[i])
plt.xlabel("Distance")
plt.ylabel("Numbers")
plt.title("Cumulative Distribution of Nearest Neighbor Distances (k=8) for OM02 Realisations")

plt.tight_layout()
plt.savefig("outfiles/nn_cumulative_distribution_OM02.png")

plt.figure()
for i in range(N):
    plt.plot(theta, kk_list[i]) 
plt.xlabel("Separation Angle (arcmin)")
plt.ylabel("Convergence Correlation")
plt.title("Convergence Correlation Function for OM02 Realisations")
plt.savefig("outfiles/kk_correlation_OM02.png")