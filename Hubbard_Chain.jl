#   Hubbard Model on a chain Lattice
#   ================================

# Import packages we need
using LinearAlgebra
using Random
using Printf

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm

# Create a function for the foldername
function build_folder_chain(sID::Int, U::Float64, μ::Float64, β::Float64, L::Int, N_burnin::Int, N_updates::Int, N_bins::Int; filepath = "data")
    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_chain_U%.2f_mu%.2f_L%d_b%.2f" U μ L β

    # Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)
    return simulation_info
end

# Initialize random number generator
function initialize_rng(seed = abs(rand(Int)))
    # seed = abs(rand(Int))
    rng = Xoshiro(seed)
    return seed,rng
end

function calculate_Lτ(β::Float64;Δτ=0.10) # Beta must be a float
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)
    return Lτ,Δτ
end

function flags(checkerboard=false, symmetric=false)
    return (checkerboard, symmetric)
end

function constants(n_stab = 10,δG_max = 1e-6)
    return (n_stab, δG_max)
end

function calculate_binsize(N_updates, N_bins)
    return div(N_updates, N_bins)
end

#   https://smoqysuite.github.io/LatticeUtilities.jl/stable/examples/#chain-Lattice

function run_hubbard_chain_simulation(sID, U, μ, β, L, N_burnin, N_updates, N_bins)
    # create the folder to save data
    simulation_info = build_folder_chain(sID, U, μ, β, L, N_burnin, N_updates, N_bins; filepath = "data")
    
    # Initialize random number generator
    seed, rng = initialize_rng()
    
    # Calculate Lτ
    Lτ,Δτ = calculate_Lτ(β,Δτ=0.10)
    
    # Get flags
    (checkerboard, symmetric) = flags()
    
    # Get constants
    (n_stab, δG_max) = constants()
    
    # Calculate bin size
    bin_size = calculate_binsize(N_updates, N_bins)
    
    # Create additional info dictionary
    additional_info = Dict(
    "dG_max" => δG_max,
    "N_burnin" => N_burnin,
    "N_updates" => N_updates,
    "N_bins" => N_bins,
    "bin_size" => bin_size,
    "local_acceptance_rate" => 0.0,
    "n_stab_init" => n_stab,
    "symmetric" => symmetric,
    "checkerboard" => checkerboard,
    "seed" => seed,
    )

    # Define model, build lattice,etc
    # Define unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0]],
        basis_vecs = [[0.0]]
    )

    # Define finite lattice with periodic boundary conditions.
    lattice = lu.Lattice(
        L = [L],
        periodic = [true]
    )

    # Initialize model geometry.
    model_geometry = ModelGeometry(
        unit_cell, lattice
    )

    # Define the nearest-neighbor bond for a 1D chain.
    bond = lu.Bond(
        orbitals = (1,1),
        displacement = [1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_id = add_bond!(model_geometry, bond)

    # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0

    # Define the non-interacting tight-binding model.
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond], # defines hopping
        t_mean  = [1.], # defines corresponding mean hopping amplitude
        t_std   = [0.], # defines corresponding standard deviation in hopping amplitude
        ϵ_mean  = [0.], # set mean on-site energy for each orbital in unit cell
        ϵ_std   = [0.], # set standard deviation of on-site energy or each orbital in unit cell
        μ       = μ # set chemical potential
    )

    hubbard_model = HubbardModel(
        shifted = false, # If true, Hubbard interaction instead parameterized as U⋅nup⋅ndn
        U_orbital = [1], #NOTE This may need to be changed? 
        U_mean = [U],
    )

    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model,)
    )

    #########################################
    ### INITIALIZE FINITE MODEL PARAMETERS ##
    #########################################

    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    hubbard_parameters = HubbardParameters(
        model_geometry = model_geometry,
        hubbard_model = hubbard_model,
        rng = rng
    )

    hubbard_ising_parameters = HubbardIsingHSParameters(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_parameters,
        rng = rng
    )

    ##############################
    ### INITIALIZE MEASUREMENTS ##
    ##############################

    # Initialize the container that measurements will be accumulated into.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    # Initialize the tight-binding model related measurements, like the hopping energy.
    initialize_measurements!(measurement_container, tight_binding_model)

    # Initialize the Hubbard interaction related measurements.
    initialize_measurements!(measurement_container, hubbard_model)

    # Initialize the single-particle electron Green's function measurement.
    # Because `time_displaced = true`, the time-displaced Greens function will be measured.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    # Initialize density correlation function measurement.
    # Because `time_displaced = false` and `integrated = true` the equal-time
    # density correlation function, and the charge susceptibility will
    # be measured. Note that the charge susceptibilty can be understood as the
    # integral of the time-displaced density correlation function over
    # the imaginary-time axis from τ=0 to τ=β.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize the pair correlation function measurement.
    # Measure the local s-wave equal-time pair correlation function (`time-displaced = false`),
    # and the corresponding pair susceptibility (`integrated = true`).
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize the spin-z correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    #############################
    ### SET-UP DQMC SIMULATION ##
    #############################
    

    # The code appearing in this section of the code is relatively boiler plate.
    # While it may change in some small ways from system to system, the overall
    # structure should remain relatively static.

    # Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the FermionPathIntegral type for both the spin-up and spin-down electrons.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_ising_parameters)

    # Initialize the imaginary-time propagators for each imaginary-time slice for both the
    # spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator for the spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    # Allcoate matrices for spin-up and spin-down electron Green's function matrices.
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    # Initialize the spin-up and spin-down electron Green's function matrices, also
    # calculating their respective determinants as the same time.
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    # Allocate matrices for various time-displaced Green's function matrices.
    Gup_ττ = similar(Gup) # G↑(τ,τ)
    Gup_τ0 = similar(Gup) # G↑(τ,0)
    Gup_0τ = similar(Gup) # G↑(0,τ)
    Gdn_ττ = similar(Gdn) # G↓(τ,τ)
    Gdn_τ0 = similar(Gdn) # G↓(τ,0)
    Gdn_0τ = similar(Gdn) # G↓(0,τ)

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization updates.
    for n in 1:N_burnin

        # Perform a sweep through the lattice, attemping an update to each Ising HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hubbard_ising_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record the acceptance rate for the attempted local updates to the HS fields.
        additional_info["local_acceptance_rate"] += acceptance_rate
    end

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    # Re-initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in 1:N_bins

        # Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            # Perform a sweep through the lattice, attemping an update to each Ising HS field.
            (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hubbard_ising_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            # Record the acceptance rate for the attempted local updates to the HS fields.
            additional_info["local_acceptance_rate"] += acceptance_rate

            # Make measurements, with the results being added to the measurement container.
            (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
                measurement_container,
                logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
                logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (hubbard_parameters, hubbard_ising_parameters)
            )
        end

        # Write the average measurements for the current bin to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

    # Calculate acceptance rate for local updates.
    additional_info["local_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator_up.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

    # Write simulation summary TOML file.
    # This simulation summary file records the version number of SmoQyDQMC and Julia
    # used to perform the simulation. The dictionary `additional_info` is appended
    # as a table to the end of the simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

   # Note that everyting appearing in this section of the code is considered post-processing,
   # and can be re-run so long as the data folder generated by the DQMC simulation persists
   # and none of the binned data has been deleted from it.

    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(simulation_info.datafolder, N_bins)

    # Merge binary files containing binned data into a single file.
    # compress_jld2_bins(folder = simulation_info.datafolder)

    # Delete the binary files containing the binned data.
    # delete_jld2_bins(folder = simulation_info.datafolder)

    return nothing

end


function get_folder_name_chain(sID::Int, U::Float64, μ::Float64, β::Float64, L::Int, N_burnin::Int, N_updates::Int, N_bins::Int; filepath = "data")
    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_chain_U%.2f_mu%.2f_L%d_b%.2f" U μ L β

    # Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID
    )

    # Initialize the directory the data will be written to.
    return simulation_info.datafolder_name
end

if abspath(PROGRAM_FILE) == @__FILE__

    # Read in the command line arguments.
    sID = parse(Int, ARGS[1]) # simulation ID
    U = parse(Float64, ARGS[2]) # energy
    μ = parse(Float64, ARGS[3]) # chemical potential
    β = parse(Float64, ARGS[4]) # inverse temperature
    L = parse(Int, ARGS[5]) # system size 
    N_burnin = parse(Int, ARGS[6]) # warm up sweeps
    N_updates = parse(Int, ARGS[7]) # measurement sweeps
    N_bins = parse(Int, ARGS[8]) #measurement bins

    # check if the simulation has already been run before
    folder_name = @sprintf "hubbard_chain_U%.2f_mu%.2f_L%d_b%.2f-%d" U μ L β sID

    println("sID: ", sID)
    println("U: ", U)
    println("μ: ", μ)
    println("β: ", β)
    println("L: ", L)
    println("N_burnin: ", N_burnin)
    println("N_updates: ", N_updates)
    println("N_bins: ", N_bins)

    println("folder_name: ", folder_name)

    # check that data folder exists
    if isdir("data")
        println("Data folder exists: data/")
    else
        println("Data folder does not exist, creating it now.")
        mkpath("data")
    end

    # check if folder/global_stats.csv exists
    if isfile("data/$folder_name/global_stats.csv")
        println("global_stats.csv exists, skipping this simulation")
        
    else
        println("global_stats.csv does not exist")
        println("Check if folder exists")
        if isdir("data/$folder_name")
            println("Folder exists, delete and start over ")
            rm("data/$folder_name", recursive=true)
        else
            println("Folder does not exist,continuing")
        # Run the simulation
        run_hubbard_chain_simulation(sID, U, μ, β, L, N_burnin, N_updates, N_bins)
        end
    end
end
