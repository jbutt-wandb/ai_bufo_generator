#!/usr/bin/env bash
# Allocate 2x H100, start the marimo app inside the allocation, and print the
# SSH tunnel command. Run this ON THE CLUSTER LOGIN NODE from ~/bufo_gen.
set -euo pipefail
PORT="${PORT:-2718}"

JOB=$(sbatch --parsable -p h100 --gres=gpu:2 --cpus-per-task=16 --mem=200G \
  --job-name=bufo-gen --wrap "sleep infinity")
echo "Allocation $JOB requested; waiting for it to start..."
while [ "$(squeue -j "$JOB" -h -o %T)" != "RUNNING" ]; do sleep 5; done
NODE=$(squeue -j "$JOB" -h -o %N)
# Compute-node hostnames don't resolve from outside the cluster fabric (and not
# even on the login node), so the tunnel must target the node's IP.
NODE_ADDR=$(scontrol show node "$NODE" | grep -oE "NodeAddr=[^ ]+" | cut -d= -f2)
echo "Allocation running on $NODE ($NODE_ADDR); starting marimo (logs/marimo.log)..."

mkdir -p logs
# uv isn't on PATH under srun, so call the venv binaries directly. The env vars
# are required for MiniMax-H3 on these nodes (see src/bufo_gen/generator.py).
nohup env TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  srun --jobid="$JOB" --overlap --export=ALL \
  .venv/bin/marimo run app.py --host 0.0.0.0 --port "$PORT" --headless \
  > logs/marimo.log 2>&1 &

cat <<EOF

From your laptop:
  ssh -N -L $PORT:$NODE_ADDR:$PORT coreweave-login
then open http://localhost:$PORT

Tear down when done:  scancel $JOB
EOF
