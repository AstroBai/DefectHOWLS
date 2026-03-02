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
    bin_edges = np.linspace(-0.04, 0.04, 50)
    df = DefectFinder(f"{OM02_DIR}/angles_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz")
    df.load_defects(f"{OM02_DIR}/defects_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz")
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
    bin_edges1 = np.logspace(np.log10(0.6), np.log10(6), 10)
    bin_edges2 = np.logspace(np.log10(1.2), np.log10(12), 10)
    bin_edges4 = np.logspace(np.log10(2.4), np.log10(24), 10)
    bin_edges8 = np.logspace(np.log10(4.8), np.log10(48), 10)
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

    fits_path = f"{OM02_DIR}/defect_catalog_s467_th_{i}_LCDM_Om02_ks_nomask_shear"
    df.save_defects(fits_path)
    theta, pp = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_+0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, nn = df.correlation_function(charge1_path=f"{fits_path}_charge_-0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, pn = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)

    df = DefectFinder(f"{OM02_DIR}/angles_{i}_LCDM_Om02_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.59, max_sep=48, n_bins=20)

    np.savez(f"{OM02_DIR}/OM02_stats_s467_th_lessbins_{i}.npz",
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

    
for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.04, 0.04, 50)
    df = DefectFinder(f"{OM04_DIR}/angles_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz")
    df.load_defects(f"{OM04_DIR}/defects_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz")
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
    bin_edges1 = np.logspace(np.log10(0.6), np.log10(6), 10)
    bin_edges2 = np.logspace(np.log10(1.2), np.log10(12), 10)
    bin_edges4 = np.logspace(np.log10(2.4), np.log10(24), 10)
    bin_edges8 = np.logspace(np.log10(4.8), np.log10(48), 10)
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

    fits_path = f"{OM04_DIR}/defect_catalog_s467_th_{i}_LCDM_Om04_ks_nomask_shear"
    df.save_defects(fits_path)
    theta, pp = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_+0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, nn = df.correlation_function(charge1_path=f"{fits_path}_charge_-0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, pn = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    

    df = DefectFinder(f"{OM04_DIR}/angles_{i}_LCDM_Om04_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.59, max_sep=48, n_bins=20)

    np.savez(f"{OM04_DIR}/OM04_stats_s467_th_lessbins_{i}.npz",
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

for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.04, 0.04, 50)
    df = DefectFinder(f"{S807_DIR}/angles_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz")
    df.load_defects(f"{S807_DIR}/defects_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz")
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
    bin_edges1 = np.logspace(np.log10(0.6), np.log10(6), 10)
    bin_edges2 = np.logspace(np.log10(1.2), np.log10(12), 10)
    bin_edges4 = np.logspace(np.log10(2.4), np.log10(24), 10)
    bin_edges8 = np.logspace(np.log10(4.8), np.log10(48), 10)
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

    fits_path = f"{S807_DIR}/defect_catalog_s467_th_{i}_LCDM_s80707210_ks_nomask_shear"
    df.save_defects(fits_path)
    theta, pp = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_+0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, nn = df.correlation_function(charge1_path=f"{fits_path}_charge_-0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, pn = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)

    df = DefectFinder(f"{S807_DIR}/angles_{i}_LCDM_s80707210_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.59, max_sep=48, n_bins=20)

    np.savez(f"{S807_DIR}/S807_stats_s467_th_lessbins_{i}.npz",
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

for i in tqdm(range(N)):
    bin_edges = np.linspace(-0.04, 0.04, 50)
    df = DefectFinder(f"{S809_DIR}/angles_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz")
    df.load_defects(f"{S809_DIR}/defects_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz")
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
    bin_edges1 = np.logspace(np.log10(0.6), np.log10(6), 10)
    bin_edges2 = np.logspace(np.log10(1.2), np.log10(12), 10)
    bin_edges4 = np.logspace(np.log10(2.4), np.log10(24), 10)
    bin_edges8 = np.logspace(np.log10(4.8), np.log10(48), 10)
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

    fits_path = f"{S809_DIR}/defect_catalog_s467_th_{i}_LCDM_s80976624_ks_nomask_shear"
    df.save_defects(fits_path)
    theta, pp = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_+0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, nn = df.correlation_function(charge1_path=f"{fits_path}_charge_-0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20)
    theta, pn = df.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                        min_sep=0.59, max_sep=48, n_bins=20) 

    df = DefectFinder(f"{S809_DIR}/angles_{i}_LCDM_s80976624_ks_nomask_shear.npz")
    theta, kk = df.convergence_correlation(min_sep=0.59, max_sep=48, n_bins=20)

    np.savez(f"{S809_DIR}/S809_stats_s467_th_lessbins_{i}.npz",
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

N = 1101
for i in tqdm(range(N)):
    try:
        # split into 4 sub-fields of 512x512 pixels each
        for subfield in range(4):
            bin_edges = np.linspace(-0.04, 0.04, 50)
            df_subfield = DefectFinder(f"{SLICS_DIR}/angles_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
            df_subfield.load_defects(f"{SLICS_DIR}/defects_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
            kappa_subfield, _ = np.histogram(df_subfield.kappa, bins=bin_edges, density=True)
            kappa_pos_subfield = df_subfield.convergence_dist(charge=0.5)
            kappa_pos_cum_subfield = np.array([np.sum(kappa_pos_subfield < b) for b in bin_edges]) / 25 * (180/np.pi)**2 
            kappa_neg_subfield = df_subfield.convergence_dist(charge=-0.5)
            kappa_neg_cum_subfield = np.array([np.sum(kappa_neg_subfield < b) for b in bin_edges]) / 25 * (180/np.pi)**2    
            nn_pos_same1 = df_subfield.nearest_distance(charge=0.5, k=1)
            nn_neg_same1 = df_subfield.nearest_distance(charge=-0.5, k=1) 
            nn_pos_opp1 = df_subfield.nearest_opposite_distance(charge=0.5, k=1)
            nn_neg_opp1 = df_subfield.nearest_opposite_distance(charge=-0.5, k=1)
            nn_pos_same2 = df_subfield.nearest_distance(charge=0.5, k=2)
            nn_neg_same2 = df_subfield.nearest_distance(charge=-0.5, k=2)
            nn_pos_opp2 = df_subfield.nearest_opposite_distance(charge=0.5, k=2)
            nn_neg_opp2 = df_subfield.nearest_opposite_distance(charge=-0.5, k=2)
            nn_pos_same4 = df_subfield.nearest_distance(charge=0.5, k=4)
            nn_neg_same4 = df_subfield.nearest_distance(charge=-0.5, k=4)
            nn_pos_opp4 = df_subfield.nearest_opposite_distance(charge=0.5, k=4)
            nn_neg_opp4 = df_subfield.nearest_opposite_distance(charge=-0.5, k=4)
            nn_pos_same8 = df_subfield.nearest_distance(charge=0.5, k=8)
            nn_neg_same8 = df_subfield.nearest_distance(charge=-0.5, k=8)
            nn_pos_opp8 = df_subfield.nearest_opposite_distance(charge=0.5, k=8)
            nn_neg_opp8 = df_subfield.nearest_opposite_distance(charge=-0.5, k=8)
            bin_edges1 = np.logspace(np.log10(0.6), np.log10(6), 10)
            bin_edges2 = np.logspace(np.log10(1.2), np.log10(12), 10)
            bin_edges4 = np.logspace(np.log10(2.4), np.log10(24), 10)
            bin_edges8 = np.logspace(np.log10(4.8), np.log10(48), 10)
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
            
            fits_path = f"{SLICS_DIR}/defect_catalog_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear"
            df_subfield.save_defects(fits_path)
            theta, pp = df_subfield.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                                        charge2_path=f"{fits_path}_charge_+0.5.fits",
                                                        min_sep=0.59, max_sep=48, n_bins=20)
            theta, nn = df_subfield.correlation_function(charge1_path=f"{fits_path}_charge_-0.5.fits",
                                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                                        min_sep=0.59, max_sep=48, n_bins=20)
            theta, pn = df_subfield.correlation_function(charge1_path=f"{fits_path}_charge_+0.5.fits",
                                                        charge2_path=f"{fits_path}_charge_-0.5.fits",
                                                        min_sep=0.59, max_sep=48, n_bins=20)

            df_subfield = DefectFinder(f"{SLICS_DIR}/angles_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
            theta, kk = df_subfield.convergence_correlation(min_sep=0.59, max_sep=48, n_bins=20)
            np.savez(f"{SLICS_DIR}/SLICS_stats_s467_th_lessbins_{i}_subfield{subfield}.npz",
                        kappa=kappa_subfield,
                        kappa_pos_cum=kappa_pos_cum_subfield,
                        kappa_neg_cum=kappa_neg_cum_subfield,
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

