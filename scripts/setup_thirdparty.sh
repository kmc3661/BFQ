#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
THIRDPARTY_DIR="$REPO_ROOT/3rdparty"
mkdir -p "$THIRDPARTY_DIR"

if [ ! -d "$THIRDPARTY_DIR/LLaVA-NeXT/.git" ]; then
  git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git "$THIRDPARTY_DIR/LLaVA-NeXT"
fi

if [ ! -d "$THIRDPARTY_DIR/lmms-eval/.git" ]; then
  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git "$THIRDPARTY_DIR/lmms-eval"
fi

echo "Third-party repositories are ready under $THIRDPARTY_DIR"
