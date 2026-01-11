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


for i in tqdm(range(N)):
    data = np.load(f"{OM02_DIR}/OM02_stats_s467_{i}.npz")
    bin_edges = data['bin_edges']
    bin_edges1 = data['bin_edges1']
    bin_edges2 = data['bin_edges2']
    bin_edges4 = data['bin_edges4']
    bin_edges8 = data['bin_edges8']
    kappa = data['kappa'][:-2]
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
    theta = data['theta']
    kk = data['kk']
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
all_mean = np.concatenate([kappa_pos_mean, kappa_neg_mean,
                           nn_pos_same1_mean, nn_neg_same1_mean, nn_pos_opp1_mean, nn_neg_opp1_mean,
                           nn_pos_same2_mean, nn_neg_same2_mean, nn_pos_opp2_mean, nn_neg_opp2_mean,
                           nn_pos_same4_mean, nn_neg_same4_mean, nn_pos_opp4_mean, nn_neg_opp4_mean,
                           nn_pos_same8_mean, nn_neg_same8_mean, nn_pos_opp8_mean, nn_neg_opp8_mean])
all_and_pdf_mean = np.concatenate([all_mean, kappa_mean])
all_howls_and_kk_mean = np.concatenate([all_and_pdf_mean, kk_mean])

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
np.savez(f"{OM02_DIR}/OM02_stats_cov_mean_s467.npz",
         bin_edges=bin_edges,
         bin_edges1=bin_edges1,
         bin_edges2=bin_edges2,
         bin_edges4=bin_edges4,
         bin_edges8=bin_edges8,
         theta=theta,
         kk_mean=kk_mean,
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
         all_mean=all_mean,
         nn_1_mean=nn_1_mean,
         nn_2_mean=nn_2_mean,
         nn_4_mean=nn_4_mean,
         nn_8_mean=nn_8_mean,
         nn_all_mean=nn_all_mean,
         all_and_pdf_mean=all_and_pdf_mean,
         all_howls_and_kk_mean=all_howls_and_kk_mean,
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
            all_howls_and_kk_cov=all_howls_and_kk_cov
         )

# =======================================================================

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


for i in tqdm(range(N)):
    data = np.load(f"{OM04_DIR}/OM04_stats_s467_{i}.npz")
    bin_edges = data['bin_edges']
    bin_edges1 = data['bin_edges1']
    bin_edges2 = data['bin_edges2']
    bin_edges4 = data['bin_edges4']
    bin_edges8 = data['bin_edges8']
    theta = data['theta']
    kk = data['kk']
    kappa = data['kappa'][:-2]
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
all_mean = np.concatenate([kappa_pos_mean, kappa_neg_mean,
                           nn_pos_same1_mean, nn_neg_same1_mean, nn_pos_opp1_mean, nn_neg_opp1_mean,
                           nn_pos_same2_mean, nn_neg_same2_mean, nn_pos_opp2_mean, nn_neg_opp2_mean,
                           nn_pos_same4_mean, nn_neg_same4_mean, nn_pos_opp4_mean, nn_neg_opp4_mean,
                           nn_pos_same8_mean, nn_neg_same8_mean, nn_pos_opp8_mean, nn_neg_opp8_mean])
all_and_pdf_mean = np.concatenate([all_mean, kappa_mean])
all_howls_and_kk_mean = np.concatenate([all_and_pdf_mean, kk_mean])
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
np.savez(f"{OM04_DIR}/OM04_stats_cov_mean_s467.npz",
         bin_edges=bin_edges,
         bin_edges1=bin_edges1,
         bin_edges2=bin_edges2,
         bin_edges4=bin_edges4,
         bin_edges8=bin_edges8,
            theta=theta,
            kk_mean=kk_mean,
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
         all_howls_and_kk_cov=all_howls_and_kk_cov
         )

# =======================================================================


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


for i in tqdm(range(N)):
    data = np.load(f"{S807_DIR}/S807_stats_s467_{i}.npz")
    bin_edges = data['bin_edges']
    bin_edges1 = data['bin_edges1']
    bin_edges2 = data['bin_edges2']
    bin_edges4 = data['bin_edges4']
    bin_edges8 = data['bin_edges8']
    theta = data['theta']
    kk = data['kk']
    kappa = data['kappa'][:-2]
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
all_mean = np.concatenate([kappa_pos_mean, kappa_neg_mean,
                           nn_pos_same1_mean, nn_neg_same1_mean, nn_pos_opp1_mean, nn_neg_opp1_mean,
                           nn_pos_same2_mean, nn_neg_same2_mean, nn_pos_opp2_mean, nn_neg_opp2_mean,
                           nn_pos_same4_mean, nn_neg_same4_mean, nn_pos_opp4_mean, nn_neg_opp4_mean,
                           nn_pos_same8_mean, nn_neg_same8_mean, nn_pos_opp8_mean, nn_neg_opp8_mean])
all_and_pdf_mean = np.concatenate([all_mean, kappa_mean])
all_howls_and_kk_mean = np.concatenate([all_and_pdf_mean, kk_mean])

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
np.savez(f"{S807_DIR}/S807_stats_cov_mean_s467.npz",
         bin_edges=bin_edges,
         bin_edges1=bin_edges1,
         bin_edges2=bin_edges2,
         bin_edges4=bin_edges4,
         bin_edges8=bin_edges8,
            theta=theta,
            kk_mean=kk_mean,
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
         all_howls_and_kk_cov=all_howls_and_kk_cov
         )

# =======================================================================

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


for i in tqdm(range(N)):
    data = np.load(f"{S809_DIR}/S809_stats_s467_{i}.npz")
    bin_edges = data['bin_edges']
    bin_edges1 = data['bin_edges1']
    bin_edges2 = data['bin_edges2']
    bin_edges4 = data['bin_edges4']
    bin_edges8 = data['bin_edges8']
    theta = data['theta']
    kk = data['kk']
    kappa = data['kappa'][:-2]
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
all_mean = np.concatenate([kappa_pos_mean, kappa_neg_mean,
                           nn_pos_same1_mean, nn_neg_same1_mean, nn_pos_opp1_mean, nn_neg_opp1_mean,
                           nn_pos_same2_mean, nn_neg_same2_mean, nn_pos_opp2_mean, nn_neg_opp2_mean,
                           nn_pos_same4_mean, nn_neg_same4_mean, nn_pos_opp4_mean, nn_neg_opp4_mean,
                           nn_pos_same8_mean, nn_neg_same8_mean, nn_pos_opp8_mean, nn_neg_opp8_mean])
all_and_pdf_mean = np.concatenate([all_mean, kappa_mean])
all_howls_and_kk_mean = np.concatenate([all_and_pdf_mean, kk_mean])

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
np.savez(f"{S809_DIR}/S809_stats_cov_mean_s467.npz",
         bin_edges=bin_edges,
         bin_edges1=bin_edges1,
         bin_edges2=bin_edges2,
         bin_edges4=bin_edges4,
         bin_edges8=bin_edges8,
            theta=theta,
            kk_mean=kk_mean,
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
         all_howls_and_kk_cov=all_howls_and_kk_cov
         )

# =======================================================================
N = 1000
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


for i in tqdm(range(N)):
    try:
        data = np.load(f"{SLICS_DIR}/SLICS_stats_s467_{i}.npz")
        bin_edges = data['bin_edges']
        bin_edges1 = data['bin_edges1']
        bin_edges2 = data['bin_edges2']
        bin_edges4 = data['bin_edges4']
        bin_edges8 = data['bin_edges8']
        theta = data['theta']
        kk = data['kk']
        kappa = data['kappa'][:-2]
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
np.savez(f"{SLICS_DIR}/SLICS_stats_cov_mean_s467.npz",
         bin_edges=bin_edges,
         bin_edges1=bin_edges1,
         bin_edges2=bin_edges2,
         bin_edges4=bin_edges4,
         bin_edges8=bin_edges8,
            theta=theta,
            kk_mean=kk_mean,
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
         all_howls_and_kk_cov=all_howls_and_kk_cov
         )