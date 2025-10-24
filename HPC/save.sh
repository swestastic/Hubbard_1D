#!/bin/bash
#SBATCH --job-name=Save

readonly Mu_i=0.0
readonly Mu_f=10.0
readonly Mu_step=0.1

readonly U_i=4.0
readonly U_f=4.0
readonly U_step=1.0

readonly L_i=50
readonly L_f=50
readonly L_step=1

readonly Beta_i=2.0
readonly Beta_f=2.0
readonly Beta_step=1.0

readonly KAGOME=True
readonly SQUARE=False

readonly sID_i=1
readonly sID_f=25
readonly sID_step=1

readonly SAVE=True

python3 save.py --Mu_i $Mu_i --Mu_f $Mu_f --Mu_step $Mu_step --U_i $U_i --U_f $U_f --U_step $U_step --L_i $L_i --L_f $L_f --L_step $L_step --Beta_i $Beta_i --Beta_f $Beta_f --Beta_step $Beta_step --KAGOME $KAGOME --SQUARE $SQUARE --sID_i $sID_i --sID_f $sID_f --sID_step $sID_step --SAVE $SAVE
