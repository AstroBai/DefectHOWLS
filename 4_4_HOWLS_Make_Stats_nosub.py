from DefectFinder import DefectFinder
from tqdm import tqdm
import numpy as np
import os

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"


N = 1101
for i in tqdm(range(N)):
    try:
        bin_edges = np.linspace(-0.04, 0.04, 100)
        df = DefectFinder(f"{SLICS_DIR}/angles_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        df.load_defects(f"{SLICS_DIR}/defects_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz")
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
        bin_edges1 = np.logspace(np.log10(0.6), np.log10(6), 20)
        bin_edges2 = np.logspace(np.log10(1.2), np.log10(12), 20)
        bin_edges4 = np.logspace(np.log10(2.4), np.log10(24), 20)
        bin_edges8 = np.logspace(np.log10(4.8), np.log10(48), 20)
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

        fits_path = f"{SLICS_DIR}/defect_catalog_s467_th_{i}_LCDM_SLICS_ks_nomask_shear"
        df.save_defects(fits_path)
        theta, pp = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                            charge2_path=f"{fits_path}_charge_+0.5.fits",
                                            min_sep=0.59, max_sep=40, n_bins=20)
        theta, nn = df.correlation_function(charge1_path=f"{fits_path}_charge_-0.5.fits",
                                            charge2_path=f"{fits_path}_charge_-0.5.fits",
                                            min_sep=0.59, max_sep=40, n_bins=20)
        theta, pn = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                            charge2_path=f"{fits_path}_charge_-0.5.fits",
                                            min_sep=0.59, max_sep=40, n_bins=20)

        df = DefectFinder(f"{SLICS_DIR}/angles_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        theta, kk = df.convergence_correlation(min_sep=0.59, max_sep=40, n_bins=20)

        np.savez(f"{SLICS_DIR}/SLICS_stats_s467_th_morebins_nosub_{i}.npz",
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
                    pp=pp,
                    nn=nn,
                    pn=pn,
                    theta=theta,
                    bin_edges=bin_edges,
                    bin_edges1=bin_edges1,
                    bin_edges2=bin_edges2,
                    bin_edges4=bin_edges4,
                    bin_edges8=bin_edges8)

    except FileNotFoundError:
        print(f"File for realisation {i} not found, skipping.")

