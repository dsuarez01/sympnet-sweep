#!/bin/bash
#SBATCH --job-name=<job name>
#SBATCH --partition=<partition name>
#SBATCH --nodes=<# nodes>
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=<# CPUs per node>
#SBATCH --mem-per-cpu=<amount mem per node>
#SBATCH --time=<time for job>
#SBATCH --exclusive
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

mkdir -p ./logs

source .venv/bin/activate

# see https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm-template.html#slurm-template
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599

# redis pw here is mostly useful for isolation, 
# e.g. if multiple users are running jobs on the same node/network
REDIS_PW=$(uuidgen)
export REDIS_PW

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# sometimes the ip addr resolution from hostname will fail
if [ -z "$ip" ]; then
  echo "ERROR: Failed to get IP address for head node $node_1"
  exit 1
fi

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

head_port=6379

ip_head=$ip:$head_port
export ip_head
echo "IP Head: $ip_head"

# set thread/proc limit
MAX_PIDS="thread/proc limit here"

# some addnl. resource calcs so Ray detects avail. resources properly
# (example calcns., adjust accordingly)
TOTAL_MEM_BYTES=$((SLURM_CPUS_PER_TASK*SLURM_MEM_PER_CPU*1024*1024))
OBJECT_STORE_BYTES=$((TOTAL_MEM_BYTES*50/100))

# redirect ray logs to tmpdir with enough space
# note also on Unix systems that there is a 107-byte
# limit on the pathname to the socket
# (what Ray appends uses roughly 2/3 of this limit):
# https://github.com/ray-project/ray/issues/55255
mkdir -p "/path/to/ray_log/tmpdir/here"

echo "MAX_PIDS (set via ulimit -u per node)=$MAX_PIDS"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_CPU=$SLURM_MEM_PER_CPU"
echo "TOTAL_MEM_BYTES=$TOTAL_MEM_BYTES"
echo "OBJECT_STORE_BYTES=$OBJECT_STORE_BYTES"

# set up head node
echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" bash -c "\
  ulimit -u "$MAX_PIDS" && \
  ray start \
  --head \
  --node-ip-address="$ip" \
  --port=$head_port \
  --num-cpus="$SLURM_CPUS_PER_TASK" \
  --memory="$TOTAL_MEM_BYTES" \
  --object-store-memory="$OBJECT_STORE_BYTES" \
  --dashboard-host="$ip" \
  --redis-password="$REDIS_PW" \
  --temp-dir="/path/to/ray_log/tmpdir/here" \
  --block
" &
sleep 30

# set up worker nodes
worker_num=$((SLURM_JOB_NUM_NODES-1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" bash -c "\
    ulimit -u "$MAX_PIDS" && \
    ray start \
    --address "$ip_head" \
    --num-cpus="$SLURM_CPUS_PER_TASK" \
    --memory="$TOTAL_MEM_BYTES" \
    --object-store-memory="$OBJECT_STORE_BYTES" \
    --redis-password="$REDIS_PW" \
    --temp-dir="/path/to/ray_log/tmpdir/here" \
    --block
  " &
  sleep 5
done

# submit to cluster
export RAY_ADDRESS="$ip_head"
ray job submit -- \
  python -u apps/run_sweep.py \
 --entity <wandb username> \
 --system <e.g. duffing> \
 --timestamp <add if resuming> \
 --enable-wandb