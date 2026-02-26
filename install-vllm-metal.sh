#!/bin/bash
# Install vllm-metal (Apple Silicon vLLM) into the current .venv
# Usage: bash install-vllm-metal.sh
#
# Why this exists:
#   vllm doesn't ship Metal/MPS support in the main package.
#   vllm-metal is a separate project that patches vllm to run on Apple Silicon GPUs.
#   The official install script creates its own venv — this one uses whatever .venv is active.
#
# What it does:
#   1. Build vllm from source (CPU-only build, no CUDA needed)
#   2. Install vllm-metal wheel on top (adds Metal GPU backend)
#
# After install, you can do:
#   from vllm import LLM
#   llm = LLM(model="sft-merged-hf")   # runs on Apple GPU

set -euo pipefail  # exit on error, undefined vars, pipe failures

# Guard: this only works on Apple Silicon (arm64 Macs)
[[ "$(uname -m)" == "arm64" ]] || { echo "Apple Silicon required"; exit 1; }

VLLM_VERSION="0.14.1"
VLLM_TARBALL="vllm-${VLLM_VERSION}.tar.gz"

# --- Step 1: Install vllm from source ---
# vllm-metal requires a specific vllm version built from source.
# We download the source tarball, install its CPU dependencies, then install vllm itself.
echo "==> Installing vllm ${VLLM_VERSION} from source..."

# Download source tarball from vllm's GitHub releases
echo "    Downloading vllm ${VLLM_VERSION} source tarball..."
curl -fsSLO "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/${VLLM_TARBALL}"
echo "    Extracting..."
tar xf "$VLLM_TARBALL"

pushd "vllm-${VLLM_VERSION}" > /dev/null
# Install vllm's CPU-only dependencies (skips CUDA/ROCm stuff)
# --index-strategy unsafe-best-match: allows mixing PyPI versions to satisfy vllm's pinned deps
echo "    Installing vllm CPU dependencies..."
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
# Build and install vllm itself
echo "    Building and installing vllm..."
uv pip install .
popd > /dev/null

# Clean up source files
echo "    Cleaning up source files..."
rm -rf "vllm-${VLLM_VERSION}" "$VLLM_TARBALL"

# --- Step 2: Install vllm-metal wheel ---
# vllm-metal publishes pre-built wheels on GitHub releases.
# We hit the GitHub API to find the latest .whl URL, then install it directly.
echo "==> Fetching latest vllm-metal release..."

# Parse the GitHub releases JSON to extract the .whl download URL
echo "    Parsing GitHub API for latest wheel URL..."
WHEEL_URL=$(curl -fsSL https://api.github.com/repos/vllm-project/vllm-metal/releases/latest \
  | python3 -c "import sys,json; assets=json.load(sys.stdin)['assets']; print(next(a['browser_download_url'] for a in assets if a['name'].endswith('.whl')))")

# uv pip can install directly from a URL — no need to download the wheel first
echo "==> Installing vllm-metal from ${WHEEL_URL##*/}..."
uv pip install "$WHEEL_URL"

echo "==> Done! Verify with: uv run python -c 'import vllm; print(vllm.__version__)'"
