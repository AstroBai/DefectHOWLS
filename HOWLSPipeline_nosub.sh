#!/bin/bash

conda init
conda activate cosmology
cd /home/jbai/DefectHOWLS

# python 0_HOWLS_Make_Data.py
# python 1_HOWLS_Angle_Map.py
# python 2_HOWLS_Defect_Catalog.py
# python 3_HOWLS_Test_Stats.py
python -u 4_4_HOWLS_Make_Stats_nosub.py
python -u 5_1_HOWLS_Make_Covs_nosub.py
python -u 6_1_HOWLS_Fisher_nosub.py