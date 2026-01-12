#!/bin/bash

conda init
conda activate cosmology
cd /home/jbai/DefectHOWLS

# python 0_HOWLS_Make_Data.py
# python 1_HOWLS_Angle_Map.py
python 2_HOWLS_Defect_Catalog.py
# python 3_HOWLS_Test_Stats.py
python 4_HOWLS_Make_Stats.py
python 5_HOWLS_Make_Covs.py
python 6_HOWLS_Fisher.py
python 7_HOWLS_Plot_Derivs.py