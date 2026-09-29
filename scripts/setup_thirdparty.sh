#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
THIRDPARTY_DIR="$REPO_ROOT/3rdparty"
LLAVA_REV=1a7e8b2b4880f7548c1883669302a8b36bd79df6
LMMS_EVAL_REV=c75f1a4e6848676b9f47c3b5df07b93f60034388
mkdir -p "$THIRDPARTY_DIR"

if [ ! -d "$THIRDPARTY_DIR/LLaVA-NeXT/.git" ]; then
  git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git "$THIRDPARTY_DIR/LLaVA-NeXT"
  git -C "$THIRDPARTY_DIR/LLaVA-NeXT" checkout --detach "$LLAVA_REV"
elif [ "$(git -C "$THIRDPARTY_DIR/LLaVA-NeXT" rev-parse HEAD)" != "$LLAVA_REV" ]; then
  echo "Warning: existing LLaVA-NeXT checkout differs from tested revision $LLAVA_REV" >&2
fi

if [ ! -d "$THIRDPARTY_DIR/lmms-eval/.git" ]; then
  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git "$THIRDPARTY_DIR/lmms-eval"
  git -C "$THIRDPARTY_DIR/lmms-eval" checkout --detach "$LMMS_EVAL_REV"
elif [ "$(git -C "$THIRDPARTY_DIR/lmms-eval" rev-parse HEAD)" != "$LMMS_EVAL_REV" ]; then
  echo "Warning: existing lmms-eval checkout differs from tested revision $LMMS_EVAL_REV" >&2
fi

echo "Third-party repositories are ready under $THIRDPARTY_DIR"
