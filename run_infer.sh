#!/bin/bash
# Inference machine launcher.
# Uses an inference-only uv environment (vLLM pins) so it never clobbers the
# training machine's torch/torchvision in the shared project folder.
set -euo pipefail
cd "$(dirname "$0")"

export UV_PROJECT_ENVIRONMENT=".venv-infer"

uv sync --extra infer          # idempotent; resolves from the shared uv.lock
uv run trl vllm-serve "$@"
