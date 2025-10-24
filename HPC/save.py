# Looks at the arguments and saves the data to a .txt file

# from juliacall import Main
import os
import numpy as np
import pandas as pd
import argparse

### argparse stuff
parser = argparse.ArgumentParser(description='Run Kagome and Square Hubbard Model Simulations')
parser.add_argument('--Mu_i', type=float, default=-8.0, help='Initial Chemical Potential')
parser.add_argument('--Mu_f', type=float, default=8.0, help='Final Chemical Potential')
parser.add_argument('--Mu_step', type=float, default=1.0, help='Step between initial and final Chemical Potentials')

parser.add_argument('--U_i', type=float, default=0.0, help='Initial Interaction Energy')
parser.add_argument('--U_f', type=float, default=8.0, help='Final Interaction Energy')
parser.add_argument('--U_step', type=float, default=1.0, help='Step between Interaction Energies')

parser.add_argument('--L_i', type=int, default=3, help='Initial L')
parser.add_argument('--L_f', type=int, default=8, help='Final L')
parser.add_argument('--L_step', type=int, default=2, help='Step betwee Ls')

parser.add_argument('--Beta_i', type=float, default=5.0, help='Initial Inverse Temperature')
parser.add_argument('--Beta_f', type=float, default=20, help='Final Inverse Temperature')
parser.add_argument('--Beta_step', type=float, default=2.0, help='Step between Inverse Temperatures')

parser.add_argument('--sID_i', type=int, default=1, help='Initial Simulation ID number')
parser.add_argument('--sID_f', type=int, default=25, help='Final Simulation ID number')
parser.add_argument('--sID_step', type=int, default=1, help='Step between Simulation ID numbers')

# parser.add_argument('--sID', type=int, default=4, help='Simulation ID number')
# parser.add_argument('--Nperbin', type=int, default=1000, help='Number of burnin steps')
# parser.add_argument('--N_updates', type=int, default=1000, help='Number of updates')
# parser.add_argument('--Nbins', type=int, default=40, help='Number of bins')

parser.add_argument('--KAGOME' , type=bool, default=True, help='Run Kagome Model')
parser.add_argument('--SQUARE' , type=bool, default=True, help='Run Square Model')

parser.add_argument('--SAVE', type=bool, default=False, help='Toggle Saving .txt output')

args = parser.parse_args()

#####################################################################################

Mus = np.arange(args.Mu_i, args.Mu_f+args.Mu_step, args.Mu_step).round(1)
Betas = np.arange(args.Beta_i, args.Beta_f+args.Beta_step, args.Beta_step).round(1)
Us = np.arange(args.U_i, args.U_f+args.U_step, args.U_step).round(1)
Ls = np.arange(args.L_i, args.L_f+args.L_step, args.L_step,dtype=int)
sIDs = np.arange(args.sID_i, args.sID_f+args.sID_step, args.sID_step)

# sID = args.sID
# Nperbin = args.Nperbin
# N_updates = args.N_updates
# Nbins = args.Nbins

KAGOME = args.KAGOME
#SQUARE = args.SQUARE
SQUARE = False
#SAVE = args.SAVE
SAVE = True

print("Mus:",Mus)
print("Betas:",Betas)
print("Us:",Us)
print("Ls:",Ls)
print("sIDs:",sIDs)

################################################### Saving data to files

def get_stats_smoqy(sID, U, mu, beta, L, filepath = "data"):
    '''
    Based on the naming parameters we supply sID, U, mu, and beta, lookup the folder name that the data from SmoQY is saved in.

    Paramters:
    sID (int): Simulation ID number
    U (float): Interaction Energy
    mu (float): Chemical Potential
    beta (float): Inverse Temperature 
    L (int): Number of times the unit cell is propogated along the unit vectors, for a 1D chain this is the number of sites
    filepath (string): Path to the parent folder where the data is stored
    
    Returns:
    global_stats (pd.dataframe): Global system statistics like density, action, compressibility, and double occupancy
    local_stats (pd.dataframe): Local statistics for each orbital like onsite energy, density, and double occupancy
    '''
    folder_name = f"hubbard_chain_U{U:.2f}_mu{mu:.2f}_L{L}_b{beta:.2f}-{sID}"
    global_stats = pd.read_csv(f"{filepath}/{folder_name}/global_stats.csv", sep='\s+')
    local_stats = pd.read_csv(f"{filepath}/{folder_name}/local_stats.csv", sep='\s+')
    return global_stats, local_stats

def get_stats_py(sID, U, mu, beta, L, filepath = "data"):
    '''
    Based on the naming parameters we supply sID, U, mu, and beta, lookup the folder name that the data from SmoQY is saved in.

    Paramters:
    sID (int): Simulation ID number
    U (float): Interaction Energy
    mu (float): Chemical Potential
    beta (float): Inverse Temperature 
    L (int): Number of times the unit cell is propogated along the unit vectors, for a 1D chain this is the number of sites
    filepath (string): Path to the parent folder where the data is stored
    
    Returns:
    global_stats (pd.dataframe): Global system statistics like density, action, compressibility, and double occupancy
    '''
    file_name = f"DQMC_Mu{mu}B{beta}U{U}N{L}-{sID}.txt"
    data = np.loadtxt(f"{filepath}/{file_name}")
    return data

def bins(data,Nperbin,Nbins):
    # Each bin is a point in the array, values are continuously added to it
    # Ex. 'Ebins' would be a 1xNbins array containing values of E (each value is a sum of Nperbin values)
    # This function takes the average of each bin
    
    # Nbins = len(data), but we can also supply it as an argument

    if Nbins != len(data):
        print('Check array size')
        return None

    # data = data.reshape(Nbins,Nperbin) # Reshape the data into a 2D array, where each row is a bin

    Bin_avgs = data / Nperbin # Not sure if this is correct but it seems roughly right? Don't really see where the math justification is though
    # You would think it should be Nperbin 

    Bin_totalavg=np.mean(Bin_avgs) #calculates one total value

    #This is where we calculate the error bars
    ErrorBars=0
    for i in range(Nbins):
        ErrorBars+=(Bin_avgs[i]-Bin_totalavg)**2
    ErrorBars=np.sqrt(1/Nbins)*np.sqrt(1/(Nbins-1))*np.sqrt(ErrorBars)

    return Bin_totalavg,ErrorBars

Nbins = 5
Nperbin = 5

################################## create empty bins

n_up_bins = np.zeros(Nbins,dtype=float)
n_down_bins = np.zeros(Nbins,dtype=float)
double_occ_bins = np.zeros(Nbins,dtype=float)

######################################################


for length in range(len(Ls)):   
    L = Ls[length]   
    for k in range(len(Us)):
        n_up_vs_T = []
        n_up_err = []
        n_down_vs_T = []
        n_down_err = []
        double_occ_vs_T = []
        double_occ_err = []

        U = Us[k]
        # print(U)
        for j in range(len(Betas)):
            beta = Betas[j]
            # print(beta)
            for i in range(len(Mus)):
                mu = Mus[i]
                # print(mu)
                if 0 > mu > -0.01:
                    mu = 0 # fix rounding errors where we see mu=-0.00 
                for sid in range(len(sIDs)):
                    sID = sIDs[sid]
                    try:
                        global_stats, local_stats = get_stats_smoqy(sID, U, mu, beta, L)
                        n_up_bins[sid%Nbins] += global_stats["MEAN_R"][8]
                        n_down_bins[sid%Nbins] += global_stats["MEAN_R"][7]
                        double_occ_bins[sid%Nbins] += global_stats["MEAN_R"][9]

                    except Exception as e:
                        print(e)
                        n_up_bins[sid%Nbins] += np.nan
                        n_down_bins[sid%Nbins] += np.nan
                        double_occ_bins[sid%Nbins] += np.nan

                n_up_vs_T_val, n_up_err_val = bins(n_up_bins, Nperbin, Nbins)  
                n_down_vs_T_val, n_down_err_val = bins(n_down_bins, Nperbin, Nbins)
                double_occ_vs_T_val, double_occ_err_val = bins(double_occ_bins, Nperbin, Nbins)
                
                n_up_vs_T.append(n_up_vs_T_val)
                n_up_err.append(n_up_err_val)
                
                n_down_vs_T.append(n_down_vs_T_val)
                n_down_err.append(n_down_err_val)
                
                double_occ_vs_T.append(double_occ_vs_T_val)
                double_occ_err.append(double_occ_err_val)

                # Reset bins for the next beta
                n_up_bins[:], n_down_bins[:], double_occ_bins[:] = 0,0,0
    
        Data = np.array([n_up_vs_T, n_up_err, n_down_vs_T, n_down_err, double_occ_vs_T, double_occ_err])
        np.savetxt(f"SmoQy_Mui{Mus[0]}Muf{Mus[-1]}Mus{args.Mu_step}B{beta}U{U}N{L}.txt", Data)
        print("SmoQy data saved successfully.")

for length in range(len(Ls)):   
    L = Ls[length]   
    for k in range(len(Us)):
        n_up_vs_T = []
        n_up_err = []
        n_down_vs_T = []
        n_down_err = []
        double_occ_vs_T = []
        double_occ_err = []

        U = Us[k]
        # print(U)
        for j in range(len(Betas)):
            beta = Betas[j]
            # print(beta)
            for i in range(len(Mus)):
                mu = -Mus[i]
                # print(mu)
                if 0 > mu > -0.01:
                    mu = 0 # fix rounding errors where we see mu=-0.00 
                for sid in range(len(sIDs)):
                    sID = sIDs[sid]
                    try:
                        data = get_stats_py(sID, U, mu, beta, L)
                        n_up_bins[sid%Nbins] += data[0]
                        n_down_bins[sid%Nbins] += data[2]
                        double_occ_bins[sid%Nbins] += data[4]

                    except Exception as e:
                        print(e)
                        n_up_bins[sid%Nbins] += np.nan
                        n_down_bins[sid%Nbins] += np.nan
                        double_occ_bins[sid%Nbins] += np.nan

                n_up_vs_T_val, n_up_err_val = bins(n_up_bins, Nperbin, Nbins)  
                n_down_vs_T_val, n_down_err_val = bins(n_down_bins, Nperbin, Nbins)
                double_occ_vs_T_val, double_occ_err_val = bins(double_occ_bins, Nperbin, Nbins)
                
                n_up_vs_T.append(n_up_vs_T_val)
                n_up_err.append(n_up_err_val)
                
                n_down_vs_T.append(n_down_vs_T_val)
                n_down_err.append(n_down_err_val)
                
                double_occ_vs_T.append(double_occ_vs_T_val)
                double_occ_err.append(double_occ_err_val)

                # Reset bins for the next beta
                n_up_bins[:], n_down_bins[:], double_occ_bins[:] = 0,0,0
    
        Data = np.array([n_up_vs_T, n_up_err, n_down_vs_T, n_down_err, double_occ_vs_T, double_occ_err])
        np.savetxt(f"DQMC_Mui{Mus[0]}Muf{Mus[-1]}Mus{args.Mu_step}B{beta}U{U}N{L}.txt", Data)
        print("DQMC data saved successfully.")