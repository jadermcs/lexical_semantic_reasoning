#!/bin/bash
# Inference machine launcher.
# Single unified env (train + vLLM) shared with the training machine; same
# dependency set, this machine just runs the vLLM rollout server.
set -euo pipefail
cd "$(dirname "$0")"

uv sync                        # idempotent; resolves from the shared uv.lock
uv run trl vllm-serve "$@"
