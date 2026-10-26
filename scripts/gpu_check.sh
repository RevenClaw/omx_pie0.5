#!/usr/bin/env bash
# Refuse to start GPU work if the A30 is not free enough.
#   scripts/gpu_check.sh train   # needs ~23 GB free (LoRA fine-tune)
#   scripts/gpu_check.sh serve   # needs ~9 GB free  (inference server)
#   scripts/gpu_check.sh <MiB>   # custom threshold
# Exit 0 = OK, 1 = not enough free memory, 2 = no GPU / nvidia-smi missing, 3 = MIG enabled.
set -euo pipefail

mode="${1:-train}"
case "$mode" in
    train) need_mib=23000 ;;
    serve) need_mib=9000 ;;
    ''|*[!0-9]*) echo "usage: $0 [train|serve|<MiB>]"; exit 2 ;;
    *) need_mib="$mode" ;;
esac

if ! command -v nvidia-smi >/dev/null; then
    echo "❌ nvidia-smi not found"; exit 2
fi

gpu="${CUDA_VISIBLE_DEVICES:-0}"; gpu="${gpu%%,*}"
IFS=',' read -r name total used mig < <(nvidia-smi -i "$gpu" \
    --query-gpu=name,memory.total,memory.used,mig.mode.current --format=csv,noheader,nounits | sed 's/, /,/g')
free=$((total - used))

echo "🖥️  GPU $gpu: $name  total=${total}MiB used=${used}MiB free=${free}MiB  MIG=$mig"

if [ "$mig" = "Enabled" ]; then
    echo "❌ MIG is enabled. A MIG slice is too small for pi0.5 LoRA; ask the admin to disable it."; exit 3
fi

apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
if [ -n "$apps" ]; then
    echo "   Processes on the GPU:"; echo "$apps" | sed 's/^/     /'
fi

if [ "$free" -lt "$need_mib" ]; then
    echo "❌ Need ${need_mib}MiB free for '$mode', have ${free}MiB. Free the GPU first."; exit 1
fi
echo "✅ Enough free memory for '$mode' (need ${need_mib}MiB)."
