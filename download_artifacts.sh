#!/bin/bash
# Download the large activation tensors + SAE checkpoints that are kept OUT of git
# (see the uncommented .gitignore lines) from the companion HuggingFace dataset,
# restoring each file to its original repo-relative path.
#
# The dataset is public, so no auth is needed. Requires `huggingface_hub`
# (pip install huggingface_hub). Override the source with ARTIFACT_REPO.
#
#   bash download_artifacts.sh
#
# Restores: activations/, activations_late/, activations_allpos/ *.safetensors
#           and sae/checkpoints*/sae_topk_32.pt
set -eu
REPO="${ARTIFACT_REPO:-nabindev3/llm-sae-difficulty-artifacts}"
PY="${PY:-python3}"
cd "$(dirname "$0")"

echo "Downloading artifacts from hf.co/datasets/$REPO ..."
"$PY" - "$REPO" <<'PYEOF'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")
repo = sys.argv[1]
# local_dir="." restores each file to its stored repo-relative path.
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=".",
                  allow_patterns=["*.safetensors", "*.pt"])
print("Artifacts restored to their repo paths.")
PYEOF
