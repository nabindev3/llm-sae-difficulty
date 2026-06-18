#!/bin/bash
# Memory-aware, resumable driver for the Gemma-2-2B + Gemma Scope extraction.
#
# Gemma-2-2B needs ~5-6GB resident; on a 16GB Mac shared with other apps, free
# RAM fluctuates and macOS jetsam kills the process when it dips. extract_gemma.py
# is resumable at chunk granularity, so this loop:
#   1. waits until enough memory is free before each attempt,
#   2. runs the extraction (which skips already-completed chunks),
#   3. re-launches after a kill, until the final output exists.
#
# Safe to leave running unattended (e.g. overnight). Override with env vars:
#   GPY=...           interpreter (default the gemma-scope venv)
#   NEED_MB=6000      free-memory threshold to attempt a run
#   MAX_ATTEMPTS=80   give up after this many cycles
set -u
GPY="${GPY:-$HOME/.venvs/gemma-scope/bin/python3}"
OUT="${OUT:-activations_gemma}"
NEED_MB="${NEED_MB:-6000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-80}"
FINAL="$OUT/gemma_sae_codes.safetensors"

avail_mb() {
  vm_stat | awk '
    /page size of/ {ps=$8}
    /Pages free/        {f=$3}
    /Pages inactive/    {i=$3}
    /Pages speculative/ {s=$3}
    /Pages purgeable/   {p=$3}
    END {gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s);gsub(/\./,"",p);
         print int((f+i+s+p)*ps/1048576)}'
}

mkdir -p "$OUT"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [ -f "$FINAL" ]; then echo "DONE: $FINAL exists"; exit 0; fi
  a=$(avail_mb)
  echo "[attempt $attempt] available ~${a}MB (need ${NEED_MB}MB)  $(date +%H:%M:%S)"
  if [ "${a:-0}" -lt "$NEED_MB" ]; then
    echo "  low memory; waiting 120s"; sleep 120; continue
  fi
  echo "  launching extraction (resumes from saved chunks)..."
  "$GPY" extract_gemma.py --layer 12 --width 16k --max_samples 5000 \
    --output_dir "$OUT" >> "$OUT/extract_gemma.log" 2>&1
  echo "  attempt exited $?"
  sleep 10
done
echo "GAVE UP after $MAX_ATTEMPTS attempts; $(ls "$OUT"/_chunks 2>/dev/null | wc -l) chunks done."
exit 1
