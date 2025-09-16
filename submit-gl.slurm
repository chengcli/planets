#!/usr/bin/env bash
#SBATCH -J jacobi-nccl
#SBATCH -A chengcli0                # <-- change if needed
#SBATCH -p spgpu                    # <-- your GPU partition/queue
#SBATCH -N 1                        # number of nodes
#SBATCH --ntasks-per-node=1         # tasks per node (ideally = GPUs per node)
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH --cpus-per-task=4           # CPU cores per task (data loading, etc.)
#SBATCH -t 00:20:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# --- Environment (adapt to your module system) ---
#module purge
#module load cuda/12.4               # or your site default
# module load nccl/2.x.x            # if NCCL is not bundled with CUDA or PyTorch
# module load python/3.11
# module load pytorch/2.3           # if you use site PyTorch; otherwise use your venv

# If using a venv:
source $HOME/pyenv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Helpful NCCL diagnostics (tune as needed)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# If your cluster needs a specific interface, uncomment and set:
# export NCCL_SOCKET_IFNAME=ib0        # or eth0/eno1, etc.

# Pick the master address from the first node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=29500

# torchrun launch parameters
NNODES=${SLURM_NNODES}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"

# Global problem size (adjust as desired)
N=4096
M=4096
ITERS=50000

# --- Launch ---
# torchrun will set RANK/LOCAL_RANK/WORLD_SIZE automatically from env when used with SLURM + srun
srun torchrun \
  --nnodes ${NNODES} \
  --nproc_per_node ${NPROC_PER_NODE} \
  --rdzv_backend=c10d \
  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
  ./run_example.py \
    --N ${N} \
    --M ${M} \
    --iters ${ITERS} \
    --print_every 20 \
    --check_every 5
