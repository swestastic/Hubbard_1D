#!/bin/bash
#SBATCH --job-name=SmoQyChain_%a
#SBATCH --output=job_%A_%a.out
#SBATCH --array=0-525

# Parameter Ranges
# for one value just set the _I and _F to the same value
# Don't set _STEP to 0 becuase it will cause a divide by zero error
BETA_I=2.0
BETA_F=2.0
BETA_STEP=1.0

U_I=4.0
U_F=4.0
U_STEP=1.0

MU_I=8.4
MU_F=10.0
MU_STEP=0.1

SID_I=1
SID_F=25
SID_STEP=1

# Calculate Number of Steps
N_BETA=$(echo "($BETA_F - $BETA_I) / $BETA_STEP + 1" | bc)
N_U=$(echo "($U_F - $U_I) / $U_STEP + 1" | bc)
N_MU=$(echo "($MU_F - $MU_I) / $MU_STEP + 1" | bc)
N_SID=$(echo "($SID_F - $SID_I) / $SID_STEP + 1" | bc)

TOTAL_JOBS=$(echo "$N_BETA * $N_U * $N_MU * $N_SID" | bc)

# Ensure the task ID does not exceed total jobs
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "Error: Task ID exceeds total jobs ($TOTAL_JOBS). Exiting."
    exit 1
fi

# Compute Indexing for Each Parameter
TASK_ID=$SLURM_ARRAY_TASK_ID

BETA_IDX=$(echo "$TASK_ID % $N_BETA" | bc)
U_IDX=$(echo "($TASK_ID / $N_BETA) % $N_U" | bc)
MU_IDX=$(echo "($TASK_ID / ($N_BETA * $N_U)) % $N_MU" | bc)
SID_IDX=$(echo "$TASK_ID / ($N_BETA * $N_U * $N_MU)" | bc)

BETA_VAL=$(printf "%.1f" $(echo "$BETA_I + $BETA_IDX * $BETA_STEP" | bc))
U_VAL=$(printf "%.1f" $(echo "$U_I + $U_IDX * $U_STEP" | bc))
MU_VAL=$(printf "%.1f" $(echo "$MU_I + $MU_IDX * $MU_STEP" | bc))
SID_VAL=$(printf "%d" $(echo "$SID_I + $SID_IDX * $SID_STEP" | bc))

# Dynamically Rename the Job
JOB_NAME="B${BETA_VAL}_U${U_VAL}_Mu${MU_VAL}_sID${SID_VAL}"
scontrol update jobid=$SLURM_JOB_ID name=$JOB_NAME

# Other parameters
readonly L=50
readonly N_burnin=2000
readonly N_updates=2000
readonly N_bins=50

# echo "Running with parameters: BETA=$BETA_VAL, U=$U_VAL, MU=$MU_VAL, SID=$SID_VAL" >> outputs.txt
julia Hubbard_Chain.jl $SID_VAL $U_VAL $MU_VAL $BETA_VAL $L $N_burnin $N_updates $N_bins