from DefectFinder import DefectFinder
from tqdm import tqdm

DATA_DIR = "/cephfs/jbai/HOWLS_Data"
OM02_DIR = f"{DATA_DIR}/LCDM_Om02_ks_nomask_shear_new"
OM04_DIR = f"{DATA_DIR}/LCDM_Om04_ks_nomask_shear_new"
S807_DIR = f"{DATA_DIR}/LCDM_s80707210_ks_nomask_shear_new"
S809_DIR = f"{DATA_DIR}/LCDM_s80976624_ks_nomask_shear_new"
SLICS_DIR = f"{DATA_DIR}/LCDM_SLICS"

# N = 128
# for i in tqdm(range(N), desc="Processing maps"):
#     df = DefectFinder(f"{OM02_DIR}/angles_{i}_LCDM_Om02_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM02_DIR}/defects_{i}_LCDM_Om02_ks_nomask_shear.npz")

#     df = DefectFinder(f"{OM04_DIR}/angles_{i}_LCDM_Om04_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM04_DIR}/defects_{i}_LCDM_Om04_ks_nomask_shear.npz")

#     df = DefectFinder(f"{S807_DIR}/angles_{i}_LCDM_s80707210_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S807_DIR}/defects_{i}_LCDM_s80707210_ks_nomask_shear.npz")

#     df = DefectFinder(f"{S809_DIR}/angles_{i}_LCDM_s80976624_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S809_DIR}/defects_{i}_LCDM_s80976624_ks_nomask_shear.npz")




# # ========================================================================================
# # do for smoothed 4.67

# N = 128
# for i in tqdm(range(N), desc="Processing maps"):
#     df = DefectFinder(f"{OM02_DIR}/angles_s467_{i}_LCDM_Om02_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM02_DIR}/defects_s467_{i}_LCDM_Om02_ks_nomask_shear.npz")

#     df = DefectFinder(f"{OM04_DIR}/angles_s467_{i}_LCDM_Om04_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM04_DIR}/defects_s467_{i}_LCDM_Om04_ks_nomask_shear.npz")
    
#     df = DefectFinder(f"{S807_DIR}/angles_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S807_DIR}/defects_s467_{i}_LCDM_s80707210_ks_nomask_shear.npz")

#     df = DefectFinder(f"{S809_DIR}/angles_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S809_DIR}/defects_s467_{i}_LCDM_s80976624_ks_nomask_shear.npz")

# N = 128
# for i in tqdm(range(N), desc="Processing maps"):
#     df = DefectFinder(f"{OM02_DIR}/angles_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM02_DIR}/defects_s467_th_{i}_LCDM_Om02_ks_nomask_shear.npz")

#     df = DefectFinder(f"{OM04_DIR}/angles_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM04_DIR}/defects_s467_th_{i}_LCDM_Om04_ks_nomask_shear.npz")
    
#     df = DefectFinder(f"{S807_DIR}/angles_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S807_DIR}/defects_s467_th_{i}_LCDM_s80707210_ks_nomask_shear.npz")

#     df = DefectFinder(f"{S809_DIR}/angles_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S809_DIR}/defects_s467_th_{i}_LCDM_s80976624_ks_nomask_shear.npz")


# N = 1101
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         df = DefectFinder(f"{SLICS_DIR}/angles_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         df.find_defects(savepath=f"{SLICS_DIR}/defects_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         # split into 4 sub-fields of 512x512 pixels each
#         for subfield in range(4):
#             df_subfield = DefectFinder(f"{SLICS_DIR}/angles_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#             df_subfield.find_defects(savepath=f"{SLICS_DIR}/defects_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")

# N = 1101
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         df = DefectFinder(f"{SLICS_DIR}/angles_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         df.find_defects(savepath=f"{SLICS_DIR}/defects_s467_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         # split into 4 sub-fields of 512x512 pixels each
#         for subfield in range(4):
#             df_subfield = DefectFinder(f"{SLICS_DIR}/angles_s467_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#             df_subfield.find_defects(savepath=f"{SLICS_DIR}/defects_s467_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")

# N = 1101
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         df = DefectFinder(f"{SLICS_DIR}/angles_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         df.find_defects(savepath=f"{SLICS_DIR}/defects_s467_th_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         # split into 4 sub-fields of 512x512 pixels each
#         for subfield in range(4):
#             df_subfield = DefectFinder(f"{SLICS_DIR}/angles_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#             df_subfield.find_defects(savepath=f"{SLICS_DIR}/defects_s467_th_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")

# N = 128
# for i in tqdm(range(N), desc="Processing maps"):
#     df = DefectFinder(f"{OM02_DIR}/angles_s234_fromshear_{i}_LCDM_Om02_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM02_DIR}/defects_s234_fromshear_{i}_LCDM_Om02_ks_nomask_shear.npz")

#     df = DefectFinder(f"{OM04_DIR}/angles_s234_fromshear_{i}_LCDM_Om04_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{OM04_DIR}/defects_s234_fromshear_{i}_LCDM_Om04_ks_nomask_shear.npz")
    
#     df = DefectFinder(f"{S807_DIR}/angles_s234_fromshear_{i}_LCDM_s80707210_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S807_DIR}/defects_s234_fromshear_{i}_LCDM_s80707210_ks_nomask_shear.npz")

#     df = DefectFinder(f"{S809_DIR}/angles_s234_fromshear_{i}_LCDM_s80976624_ks_nomask_shear.npz")
#     df.find_defects(savepath=f"{S809_DIR}/defects_s234_fromshear_{i}_LCDM_s80976624_ks_nomask_shear.npz")

# N = 1101
# for i in tqdm(range(N), desc="Processing realisations"):
#     try:
#         df = DefectFinder(f"{SLICS_DIR}/angles_s234_fromshear_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         df.find_defects(savepath=f"{SLICS_DIR}/defects_s234_fromshear_{i}_LCDM_SLICS_ks_nomask_shear.npz")
#         # split into 4 sub-fields of 512x512 pixels each
#         for subfield in range(4):
#             df_subfield = DefectFinder(f"{SLICS_DIR}/angles_s234_fromshear_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#             df_subfield.find_defects(savepath=f"{SLICS_DIR}/defects_s234_fromshear_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
#     except FileNotFoundError:
#         print(f"File for realisation {i} not found, skipping.")


N = 128
for i in tqdm(range(N), desc="Processing maps"):
    df = DefectFinder(f"{OM02_DIR}/angles_s234_gaussian_fromshear_{i}_LCDM_Om02_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{OM02_DIR}/defects_s234_gaussian_fromshear_{i}_LCDM_Om02_ks_nomask_shear.npz")

    df = DefectFinder(f"{OM04_DIR}/angles_s234_gaussian_fromshear_{i}_LCDM_Om04_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{OM04_DIR}/defects_s234_gaussian_fromshear_{i}_LCDM_Om04_ks_nomask_shear.npz")
    
    df = DefectFinder(f"{S807_DIR}/angles_s234_gaussian_fromshear_{i}_LCDM_s80707210_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{S807_DIR}/defects_s234_gaussian_fromshear_{i}_LCDM_s80707210_ks_nomask_shear.npz")

    df = DefectFinder(f"{S809_DIR}/angles_s234_gaussian_fromshear_{i}_LCDM_s80976624_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{S809_DIR}/defects_s234_gaussian_fromshear_{i}_LCDM_s80976624_ks_nomask_shear.npz")

N = 1101
for i in tqdm(range(N), desc="Processing realisations"):
    try:
        df = DefectFinder(f"{SLICS_DIR}/angles_s234_gaussian_fromshear_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        df.find_defects(savepath=f"{SLICS_DIR}/defects_s234_gaussian_fromshear_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        # split into 4 sub-fields of 512x512 pixels each
        for subfield in range(4):
            df_subfield = DefectFinder(f"{SLICS_DIR}/angles_s234_gaussian_fromshear_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
            df_subfield.find_defects(savepath=f"{SLICS_DIR}/defects_s234_gaussian_fromshear_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
    except FileNotFoundError:
        print(f"File for realisation {i} not found, skipping.")



N = 128
for i in tqdm(range(N), desc="Processing maps"):
    df = DefectFinder(f"{OM02_DIR}/angles_s467_th_fromshear_{i}_LCDM_Om02_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{OM02_DIR}/defects_s467_th_fromshear_{i}_LCDM_Om02_ks_nomask_shear.npz")

    df = DefectFinder(f"{OM04_DIR}/angles_s467_th_fromshear_{i}_LCDM_Om04_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{OM04_DIR}/defects_s467_th_fromshear_{i}_LCDM_Om04_ks_nomask_shear.npz")
    
    df = DefectFinder(f"{S807_DIR}/angles_s467_th_fromshear_{i}_LCDM_s80707210_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{S807_DIR}/defects_s467_th_fromshear_{i}_LCDM_s80707210_ks_nomask_shear.npz")

    df = DefectFinder(f"{S809_DIR}/angles_s467_th_fromshear_{i}_LCDM_s80976624_ks_nomask_shear.npz")
    df.find_defects(savepath=f"{S809_DIR}/defects_s467_th_fromshear_{i}_LCDM_s80976624_ks_nomask_shear.npz")

N = 1101
for i in tqdm(range(N), desc="Processing realisations"):
    try:
        df = DefectFinder(f"{SLICS_DIR}/angles_s467_th_fromshear_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        df.find_defects(savepath=f"{SLICS_DIR}/defects_s467_th_fromshear_{i}_LCDM_SLICS_ks_nomask_shear.npz")
        # split into 4 sub-fields of 512x512 pixels each
        for subfield in range(4):
            df_subfield = DefectFinder(f"{SLICS_DIR}/angles_s467_th_fromshear_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
            df_subfield.find_defects(savepath=f"{SLICS_DIR}/defects_s467_th_fromshear_{i}_subfield{subfield}_LCDM_SLICS_ks_nomask_shear.npz")
    except FileNotFoundError:
        print(f"File for realisation {i} not found, skipping.")
