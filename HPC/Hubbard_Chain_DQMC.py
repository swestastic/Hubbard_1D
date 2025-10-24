#!/usr/bin/env python
# coding: utf-8

# ### The Hamiltonian

# The Hamiltonian of the Fermi-Hubbard model is described by $\hat{H}=\hat{K}+\hat{V}$ 
# 
# where $\hat{K} = -\sum_{ij\sigma}t_{ij}(c_{i\sigma}^\dagger c_{j\sigma} + c_{j\sigma}^\dagger c_{i\sigma}) -\mu \sum_{i\sigma}n_{i\sigma}$, 
# 
# and $\hat{V} = U\sum_i (n_{i\uparrow}-\frac{1}{2})(n_{i\downarrow}-\frac{1}{2})$
# 
# We are simulating a 1D periodic chain with N sites. We are using the inverse temperature, $\beta=\frac{1}{T} = L\Delta \tau$ which is divided into L "imaginary time intervals" with an interval length of $\Delta \tau$. We also want to make sure that $tU(\Delta\tau)^2<\frac{1}{10}$, or $\Delta\tau<\frac{1}{\sqrt{10tU}}$. 


import numpy as np # For arrays and math calculations
import matplotlib.pyplot as plt # For plotting
import numba # for JIT compilation
import pandas as pd # For loading CSV files from SmoQy
import os # For checking if SmoQyDQMC folders exist so we can skip regenerating data 
import argparse # For command line arguments

# Set up command line arguments
parser = argparse.ArgumentParser(description='Run DQMC for the Hubbard model.')
parser.add_argument('--N', type=int, default=20, help='Number of sites on the chain (default: 20)')
parser.add_argument('--t', type=float, default=1.0, help='Hopping parameter (default: 1.0)')
parser.add_argument('--U', type=float, default=4.0, help='On-site interaction energy (default: 4.0)')
parser.add_argument('--beta', type=float, default=2.0, help='Inverse temperature beta (default: 2.0)')
parser.add_argument('--delta_tau', type=float, default=0.1, help='Imaginary time interval (default: 0.1)')
parser.add_argument('--Mu', type=float, default=0.0, help='Chemical potential (default: 0.0)')
parser.add_argument('--sID', type=int, default=1, help='Simulation ID number (for saving data)')

args = parser.parse_args()

N = args.N
t = args.t  # Hopping parameter
U = args.U  # On-site interaction energy
beta = args.beta  # Inverse temperature beta
delta_tau = args.delta_tau  # Imaginary time interval
mu = -args.Mu  # Chemical potential
sID = args.sID  # Simulation ID number for saving data
L = int(beta//delta_tau) # Number of time slices

# ### The Kinetic Energy Matrix

# First we begin by calculating the $\hat{K}$ matrix. We expect to see $\Delta\tau$ multiplying the entire matrix, and in the matrix we should see $-\mu$ on the diagonal and a $-t$ to the left and right of every $\mu$

@numba.njit
def generate_kinetic_energy_matrix(N, t, mu, delta_tau):
    """
    Generate the kinetic energy matrix for a 1D Hubbard model with periodic boundary conditions.

    Parameters:
    N (int): Number of lattice sites
    t (float): Hopping parameter
    mu (float): Chemical potential
    delta_tau (float): Discretization step

    Returns:
    numpy.ndarray: The kinetic energy matrix.
    """
    # Initialize the matrix
    k_matrix = np.zeros((N,N))

    # Fill the matrix
    for i in range(N):
        # Diagonal term: Chemical potential
        k_matrix[i, i] = -mu

        # Off-diagonal terms: Nearest-neighbor hopping
        k_matrix[(i+1)%N, i] = -t  # Hopping to the right
        k_matrix[(i-1)%N, i] = -t  # Hopping to the left

    # Scale the matrix by delta_tau
    k_matrix *= delta_tau

    return k_matrix

# ### The Interaction Energy Matrix 

# Next we can create the Interaction Energy Matrix. We first want to create an $NxL$ array $s(i,l)$ filled randomly with $\pm1$ which represents the Hubbard-Stratonovich field 
# 
# Then we make a series of L $NxN$ diagonal matrices. These will be $v_\uparrow(l=0),v_\uparrow(l=1)...v_\uparrow(l=L)$. We also have matrices for spin-down electrons where $v_\downarrow(l) = -v_\uparrow(l)$. Each $v$ matrix is multiplied by a factor of $\lambda$, given by $cosh(\lambda) = e^{\frac{U\Delta\tau}{2}}$

@numba.njit
def generate_interaction_energy_matrix(N, L, U, delta_tau, s=None):
    """
    Generate the interaction energy matrices v_up(l) and v_down(l)
    
    Parameters:
        N (int): Number of sites (dimension of the matrix)
        L (int): Number of time slices
        U (float): Interaction energy
        delta_tau (float): Discretization parameter
    
    Returns:
        tuple: Two 3D arrays (v_up, v_down), s matrix, and varlambda
    """

    # Create s(i,l) matrix
    if s is None:
        # Generate -1 and 1 values using randint
        s = np.random.randint(0, 2, size=(N, L)) * 2 - 1

    # Calculate lambda
    varlambda = np.arccosh(np.exp(U * delta_tau / 2))

    # Preallocate v_ups and v_downs
    v_ups = np.zeros((L, N, N))
    v_downs = np.zeros((L, N, N))

    # Fill v_ups and v_downs
    for l in range(L):
        for i in range(N):
            v_ups[l, i, i] = varlambda * s[i, l]
            v_downs[l, i, i] = -varlambda * s[i, l]

    return v_ups, v_downs, s, varlambda

# ### Initializing the Green's Function

# To initialize the Green's Function we need to compute it for up and down:
# 
# $G_\sigma = [ I +e^ke^{v_\sigma(1)}e^ke^{v_\sigma(2)}....e^ke^{v_\sigma(L)}]^{-1}$
# 
# $G_\sigma = [ I +e^{kL}e^{v_\sigma(1)+v_\sigma(2)+...v_\sigma(L)}]^{-1}$
# 
# k is diagonalized so we can pull it out and multiply by L

@numba.njit
def generate_greens_function_matrix(k_matrix, v_ups, v_downs, L, N):
    """
    Generate the Green's function matrix G(i,j) for a 1D Hubbard model with periodic boundary conditions

    Parameters:
    k_matrix (numpy.ndarray): The kinetic energy matrix
    v_ups (numpy.ndarray): Array of the interaction energy matrices for spin up
    v_downs (numpy.ndarray): Array of the interaction energy matrices for spin down
    L (int): Number of imaginary time slices

    Returns:
    tuple: Two lists of matrices (v_up, v_down), each with L matrices of size NxN representing the up and down Green's functions matrices
    """
    
    I = np.eye(N) # Identity matrix
    e_kL = np.exp(L*k_matrix) # Exponential of the kinetic energy matrix L times

    up_array = np.zeros_like(v_ups[0]) # N x N
    down_array = np.zeros_like(v_downs[0])
    for i in range(L):
        up_array += v_ups[i]
        down_array += v_downs[i]

    e_up = np.diag(np.diag(np.exp(up_array)))
    e_down = np.diag(np.diag(np.exp(down_array)))

    G_up = np.linalg.solve(I + e_kL @ e_up, I)
    G_down = np.linalg.solve(I + e_kL @ e_down, I)

    return G_up, G_down

# ### Updating the Green's Function

# We want to suggest a change in the Hubbard-Stratonovich field on site $i=1$ of imaginary time slice $l=L$
# 
# First we compute $d$:
# 
# $d_\uparrow = 1 + (1-[G_{\uparrow}]_{ii})(e^{-2\lambda s(i,l)}-1)$
# 
# $d_\downarrow = 1 + (1-[G_{\downarrow}]_{ii})(e^{+2\lambda s(i,l)}-1)$
# 
# $d = d_\uparrow d_\downarrow$
# 
# Then we draw a random number $0<r<1$. If $r<d$ then we set $s(i,l) = -s(i,l)$ 

def update_greens_function(G_up,G_down,k_matrix,v_ups,v_downs,varlambda,s,i,l,N):
    """
    Update the Green's function matrix G(i,j) for a 1D Hubbard model with periodic boundary conditions

    Parameters:
    G_up (numpy.ndarray): The Green's function matrix for spin up
    G_down (numpy.ndarray): The Green's function matrix for spin down
    k_matrix (numpy.ndarray): The kinetic energy matrix
    v_ups (numpy.ndarray): Array of the interaction energy matrices for spin up
    v_downs (numpy.ndarray): Array of the interaction energy matrices for spin down
    i (int): Site index
    l (int): Time slice index

    Returns:
    tuple: Two lists of matrices (G_up, G_down) each of size NxN representing the up and down Green's functions matrices
    """

    ### Propose a change to the Hubbard-Stratonovich field s(i,l)

    # First we calculate d_up, d_down
    d_up = 1 + (1-G_up[i,i])*(np.exp(-2*varlambda*s[i,l])-1)
    d_down = 1 + (1-G_down[i,i])*(np.exp(2*varlambda*s[i,l])-1)
    d = d_up*d_down

    # Draw a random number, if it is smaller than d, accept the move
    if np.random.rand() < d:
        s[i,l] *= -1 # Flip the sign of s(i,l)

        ### Update the Green's function matrix if s was updated 
        ### Note: This is the slower method O(N^3), a faster O(N^2) method is mentioned in the notes.
        v_ups, v_downs, s, varlambda = generate_interaction_energy_matrix(N, L, U, delta_tau, s=s) # Calculate new energy interaction matrices based on new s
        G_up, G_down = generate_greens_function_matrix(k_matrix, v_ups, v_downs,L,N) # calculate new Green's functions based on new v_ups and v_downs

    return G_up, G_down, s, v_ups, v_downs

# print("v_ups")
# print(v_ups)

# print("v_downs")
# print(v_downs)

# ### Wrapping the Green's Function

# We iterate through $l$ for each $i$. After completion of one loop of $i$ s, we change the Green's functions as follows:
# 
# $G_\sigma = [e^k e^{v_\sigma(l)}] G_\sigma [e^k e^{v_\sigma (l)}]^{-1}$
# 
# We use np.linalg.inv() to find the inverse, and @ for the matrix algebra. 


@numba.njit
def wrap_greens_function(k_matrix, v_ups, v_downs, l, G_up, G_down):
    I = np.eye(N)
    
    e_kl_up = np.exp(k_matrix + v_ups[l])
    e_kl_down = np.exp(k_matrix + v_downs[l])
    
    G_up =  e_kl_up @ G_up @ np.linalg.solve(e_kl_up, I)
    G_down = e_kl_down @ G_down @ np.linalg.solve(e_kl_down, I)
    return G_up, G_down

# ### Measurements

# Next we measure the observables using the following equations
# 
# **Density of electrons of spin $\sigma$ on site $i$:** $\langle n_{i\sigma} \rangle = 1- [G_\sigma]_{ii}$
# 
# **Double occupancy rate on site $i$:** $\langle n_{i\uparrow}n_{i\downarrow} \rangle = (1-[G_\uparrow]_{ii})(1-[G_\downarrow]_{ii})$
# 
# **Local moment on site $i$:** $\langle (n_{i\uparrow} - n_{i\downarrow})^2 \rangle = \langle n_{i\uparrow} + n_{i\downarrow} \rangle - 2\langle n_{i\uparrow} n_{i\downarrow} \rangle$ 
# 
# **Correlation between moments on sites $i,j$ where $i\neq j$**: 
# 
# $S_{+i} = c_{i\uparrow}^\dagger c_{i\downarrow}$, 
# 
# $S_{-j} = c_{j\downarrow}^\dagger c_{j\uparrow}$,
# 
# $\langle S_{+i}S_{-j} \rangle = -[G_\uparrow]_{ji} [G_\downarrow]_{ij}$
# 
# We will also set up bins so that we can calculate error bars.

def calculate_observables(G_up, G_down):
    """
    Calculate the observables for a 1D Hubbard model with periodic boundary conditions

    Parameters:
    G_up (numpy.ndarray): The Green's function matrix for spin up
    G_down (numpy.ndarray): The Green's function matrix for spin down

    Returns:
    tuple: The average density per site n, average double occupancy rate per site db_occ, average local moment per site, and moment correlation
    """

    # Calculate the density
    densities_up = np.zeros(N)
    densities_down = np.zeros(N)
    for i in range(N):
        densities_up[i] = 1-G_up[i,i]
        densities_down[i] = 1-G_down[i,i]
    n_up = np.mean(densities_up)
    n_down = np.mean(densities_down)

    # Calculate the double occupancy rate
    double_occ = n_up*n_down
    # double_occ = np.mean(densities_up*densities_down)

    # Calculate the local moment
    local_moment = (n_up + n_down) - 2*(n_up*n_down)

    # Calculate the moment correlation

    return n_up, n_down, double_occ, local_moment

@numba.njit
def bins(data,Nperbin,Nbins):
    '''
    Take data from a bin and calculate the average and error bars

    Paramters:
    data (numpy.ndarray): The data to be binned
    Nperbin (int): Number of measurements per bin
    Nbins (int): Number of bins
    
    Returns:
    tuple: The average (float) and error bar (float) of the data
    '''

    if Nbins != len(data):
        print('Check array size')
        return

    Bin_avgs = data / Nperbin # normalize by number of measurements per bin

    Bin_totalavg=np.mean(Bin_avgs) # calculate the average across bins

    #This is where we calculate the error bars
    ErrorBars=0
    for i in range(Nbins):
        ErrorBars+=(Bin_avgs[i]-Bin_totalavg)**2
    ErrorBars=np.sqrt(1/Nbins)*np.sqrt(1/(Nbins-1))*np.sqrt(ErrorBars)

    return Bin_totalavg,ErrorBars

# ### Full Monte Carlo

# First We have to implement $\beta = L\Delta\tau$ so we can change the temperature of the system by varying L. 
# 
# Then we have to use the method that we've used above for the Monte Carlo

wusweeps = 5000 # Number of warm-up sweeps
msweeps = 5000 # Number of measurement sweeps
skip = 0 # Skip measurements to avoid autocorrelation
bincount = 0 # Counter for binning
Nperbin = 50 # Number of measurements per bin
Nbins=msweeps//(Nperbin*(skip+1))

# Create empty bins for the observables
n_up_bins = np.zeros(Nbins)
n_down_bins = np.zeros(Nbins)
double_occ_bins = np.zeros(Nbins)
local_moment_bins = np.zeros(Nbins)

# Create arrays for observables vs temperature
# n_up_vs_T = np.zeros_like(Mus)
# n_down_vs_T = np.zeros_like(Mus)
# double_occ_vs_T = np.zeros_like(Mus)
# local_moment_vs_T = np.zeros_like(Mus)

# Create arrays for error bars
# n_up_err = np.zeros_like(Mus)
# n_down_err = np.zeros_like(Mus)
# double_occ_err = np.zeros_like(Mus)
# local_moment_err = np.zeros_like(Mus)

if os.path.exists("data") == False: # Check if the data folder exists
    os.makedirs("data") # Create the data folder if it doesn't exist


# Check if Mus are valid for the given delta_tau
if delta_tau > 1/((10*t*U)**(1/2)):
    print("Delta Tau is too large for the given U and t. Please choose a smaller delta tau.")
    print("Delta Tau: ",delta_tau)
    print("Should be less than: ", 1/((10*t*U)**(1/2)))
else:
    if os.path.exists(f"data/DQMC_Mu{mu}B{beta}U{U}N{N}.txt"):
        print(f"Data for Mu={mu}, Beta={beta}, U={U}, N={N} already exists. Skipping simulation.")
    else:
        print(f' Running simulation for Mu={mu}, Beta={beta}, U={U}, N={N}...')
        # run the simulation
        print("Mu: ", mu)
        # n_up_bins[:], n_down_bins[:], double_occ_bins[:], local_moment_bins[:] = 0, 0, 0, 0 # Empty bins

        # Generate the initial matrices for Kinetic Energy and Interaction Energy
        k_matrix = generate_kinetic_energy_matrix(N, t, mu, delta_tau) # if we're varying beta we don't need this, but for mu we do
        k_eigvals, k_eigvecs = np.linalg.eigh(k_matrix)
        k_matrix = np.diag(k_eigvals)
        # k_matrix = np.linalg.solve(k_eigvecs,np.eye(N)) @ k_matrix @ k_eigvecs # diagonalize the matrix so we can exponentiate it using np.exp() instead of scipy.linalg.expm()
        # k_matrix = np.diag(np.diag(k_matrix)) # remove off-diagonal elements since they should be zero, but sometimes we get rounding errors

        L = int(beta//delta_tau)
        v_ups, v_downs, s, varlambda = generate_interaction_energy_matrix(N, L, U, delta_tau, s=None)

        for j in range(wusweeps): # warm-up sweeps
            v_ups, v_downs, s, varlambda = generate_interaction_energy_matrix(N, L, U, delta_tau, s=s)
            G_up, G_down = generate_greens_function_matrix(k_matrix, v_ups, v_downs,L,N)

            for l in range(L-1,-1,-1): 
                for i in range(N):
                    G_up, G_down, s, v_ups, v_downs = update_greens_function(G_up, G_down, k_matrix, v_ups, v_downs, varlambda, s, i, l, N)
                G_up, G_down = wrap_greens_function(k_matrix, v_ups, v_downs, l, G_up, G_down)

        for j in range(msweeps*(1+skip)): # Measurement sweeps
            v_ups, v_downs, s, varlambda = generate_interaction_energy_matrix(N, L, U, delta_tau, s=s)
            G_up, G_down = generate_greens_function_matrix(k_matrix, v_ups, v_downs,L,N)

            for l in range(L-1,-1,-1):
                for i in range(N):
                    G_up, G_down, s, v_ups, v_downs = update_greens_function(G_up, G_down, k_matrix, v_ups, v_downs, varlambda, s, i, l, N)
                G_up, G_down = wrap_greens_function(k_matrix, v_ups, v_downs, l, G_up, G_down)

            if skip == 0 or j%skip == 0: # Skip measurements to avoid autocorrelation
                n_up, n_down, double_occ, local_moment = calculate_observables(G_up, G_down)
                n_up_bins[bincount] += n_up
                n_down_bins[bincount] += n_down
                double_occ_bins[bincount] += double_occ
                local_moment_bins[bincount] += local_moment

                bincount = (bincount+1)%Nbins

        # Process bins 
        n_up_vs_T, n_up_err = bins(n_up_bins,Nperbin,Nbins)
        n_down_vs_T, n_down_err = bins(n_down_bins,Nperbin,Nbins)
        double_occ_vs_T, double_occ_err = bins(double_occ_bins,Nperbin,Nbins)
        local_moment_vs_T, local_moment_err = bins(local_moment_bins,Nperbin,Nbins)

        output_data = np.array([n_up_vs_T, n_up_err, n_down_vs_T, n_down_err, double_occ_vs_T, double_occ_err, local_moment_vs_T, local_moment_err])
        np.savetxt(f"data/DQMC_Mu{mu}B{beta}U{U}N{N}-{sID}.txt", output_data)
        print(f"Data for Mu={mu}, Beta={beta}, U={U}, N={N}, sID={sID} saved successfully!")