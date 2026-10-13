#!/usr/bin/env bash
# Open an SSH tunnel to the running bufo-gen marimo app. Run from your laptop,
# then browse to http://localhost:2718. Ctrl+C closes the tunnel (the app keeps
# running on the cluster).
set -euo pipefail
PORT="${PORT:-2718}"

# Find the compute node of the bufo-gen allocation (the node can change between
# launches). Short node names aren't DNS-resolvable from the login node, so
# forward to the node's IP (NodeAddr) instead.
NODE=$(ssh coreweave-login 'squeue -u $USER -h -n bufo-gen -o %N' | head -1)
if [ -z "$NODE" ]; then
  echo "No bufo-gen allocation is running. Start one with:"
  echo "  ssh coreweave-login 'cd bufo_gen && ./scripts/launch_cluster.sh'"
  exit 1
fi
ADDR=$(ssh coreweave-login "scontrol show node $NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)

echo "Tunneling localhost:$PORT -> $NODE ($ADDR):$PORT ... open http://localhost:$PORT"
exec ssh -N -L "$PORT:$ADDR:$PORT" coreweave-login
