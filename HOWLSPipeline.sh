#!/bin/bash

conda init
conda activate cosmology
cd /home/jbai/DefectHOWLS

# python 0_HOWLS_Make_Data.py
# python 1_HOWLS_Angle_Map.py
# python 2_HOWLS_Defect_Catalog.py
# python 3_HOWLS_Test_Stats.py
python -u 4_3_HOWLS_Make_Stats_morebins.py
python -u 5_HOWLS_Make_Covs.py
python -u 6_HOWLS_Fisher.py
python -u 7_HOWLS_Plot_Derivs.py
python -u 9_HOWLS_Plot_Corr_Matrix.py
python -u 10_HOWLS_Chi2_Dist.py