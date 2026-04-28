# Eagle3 / DFlash on Nemotron Verifiers — Playbook

A step-by-step guide for training and serving an Eagle3 (or DFlash) drafter
against an NVIDIA **Nemotron** or **Nemotron-H** verifier using
[`vllm-project/speculators`](https://github.com/vllm-project/speculators)
v0.5.0+ and the Eagle3 hooks added to `vllm/model_executor/models/nemotron.py`
and `vllm/model_executor/models/nemotron_h.py`.

> Status: experimental. Eagle3/DFlash drafters have not been publicly validated
> on hybrid Mamba-2 + MoE verifiers like `Nemotron-3-Nano-30B-A3B`. Treat
> acceptance numbers as a measurement, not an assumption.

## What this enables

| Verifier family | vLLM model file | Spec method | Status |
| --- | --- | --- | --- |
| Plain Nemotron (Llama-style: LayerNorm+1, sq-ReLU, no `gate_proj`) | `vllm/model_executor/models/nemotron.py` | Eagle3 / DFlash via speculators | enabled by this PR |
| Nemotron-H (hybrid Mamba-2 + MoE + Attention) | `vllm/model_executor/models/nemotron_h.py` | Eagle3 / DFlash via speculators | enabled by this PR (attention-only aux taps) |
| Nemotron-H | same | MTP via `NemotronHMTPModel` | requires checkpoint to ship `mtp.*` weights — `Nemotron-3-Nano-30B-A3B` does not |
| Nemotron-NAS | `vllm/model_executor/models/nemotron_nas.py` | Eagle3 | not yet wired (apply the same pattern if needed) |

The drafter itself is the existing `Eagle3LlamaForCausalLM` registered in
[`vllm/model_executor/models/registry.py`](../../../vllm/model_executor/models/registry.py)
— no new drafter class is needed because Eagle3 drafters are decoupled from
verifier architecture (they only need the verifier's `hidden_size` /
`num_attention_heads` / `num_key_value_heads`).

## How the Nemotron-H aux-layer policy works

`hybrid_override_pattern` encodes layer types: `M`=Mamba-2, `E`=MoE/MLP,
`*`=Attention. Mamba-2 outputs depend on SSM state that the Llama-shape
Eagle3 drafter cannot continue, so we expose **only attention layers** as
candidate aux taps. The default heuristic in
`NemotronHForCausalLM.get_eagle3_default_aux_hidden_state_layers` picks three
attention indices (early / mid / late):

```python
attn_ids = tuple(i for i, c in enumerate(cfg.hybrid_override_pattern) if c == "*")
# Nemotron-3-Nano-30B-A3B → attn_ids == (5, 12, 19, 26, 33, 42)
# default returns               → (12, 26, 42)
```

You can override this per-run by supplying
`speculators_config.eagle_aux_hidden_state_layer_ids` in your speculator
config, or by calling `model.set_aux_hidden_state_layers((...))` directly.

## Prerequisites

```bash
# In your vllm clone:
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv yet
uv venv --python 3.12
source .venv/bin/activate

uv pip install -r requirements/lint.txt
pre-commit install

# Build vllm with the Eagle3 patches included (this PR's branch):
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# In a separate clone of vllm-project/speculators (v0.5.0+):
cd /path/to/speculators
uv pip install -e .
```

## Step 1 — Sanity-check the Eagle3 hooks

```bash
.venv/bin/python -m pytest tests/v1/spec_decode/test_speculators_eagle3.py -v
.venv/bin/python -m pytest tests/v1/e2e/spec_decode/test_spec_decode.py -v -k eagle3
```

These exercise the existing Eagle3 path; they should still pass with the
patched `nemotron.py` / `nemotron_h.py`. They do not yet cover Nemotron
verifiers — extend them once you have a checkpoint.

Quick interactive smoke check that hooks are installed:

```python
from transformers import AutoConfig
from vllm import LLM

llm = LLM(model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
          trust_remote_code=True, enforce_eager=True)
m = llm.llm_engine.model_executor.driver_worker.model_runner.model
assert hasattr(m, "set_aux_hidden_state_layers"), "Eagle3 hooks not installed"
assert hasattr(m, "get_eagle3_default_aux_hidden_state_layers")
print("default aux layers:", m.get_eagle3_default_aux_hidden_state_layers())
# expected for Nemotron-3-Nano-30B-A3B: (12, 26, 42)
```

## Step 2 — Speculators training script

Save under your **speculators** clone as
`examples/train/eagle3_nemotron3_nano_30b_a3b_online.sh`:

```bash
#!/bin/bash
# Online Eagle3 training for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (BF16 / FP8).
#
# Verifier: hybrid Mamba-2 (23 M) + MoE (23 E, 128+1 experts, top-6) + 6 GQA
#           attention layers (the "*" positions in hybrid_override_pattern).
# Drafter:  default Llama-shape Eagle3 head (single transformer block matching
#           verifier hidden_size=2688). Inference path in vLLM is the existing
#           Eagle3LlamaForCausalLM registry entry — no new drafter class.

set -euo pipefail

# ============ Configuration ============
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
# MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

DATASET="sharegpt"
OUTPUT_DIR="./output_nemotron3_nano_30b_a3b_eagle3"
VLLM_PORT=8000

DRAFT_VOCAB_SIZE=32000
MAX_SAMPLES=5000           # bump to >=200k for production drafters
SEQ_LENGTH=8192
EPOCHS=5
LR=1e-4

VLLM_GPUS="0,1,2,3"
VLLM_TP=2
VLLM_DP=2
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4
# =======================================

echo "=== Step 1: Prepare data ==="
.venv/bin/python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

echo "=== Step 2: Launch vLLM verifier ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" .venv/bin/python scripts/launch_vllm.py "$MODEL" \
    -- --tensor-parallel-size "$VLLM_TP" \
       --data-parallel-size "$VLLM_DP" \
       --port "$VLLM_PORT" \
       --max-model-len "$SEQ_LENGTH" \
       --enforce-eager \
       --trust-remote-code &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM (pid=$VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM on :${VLLM_PORT} ..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM ready."

echo "=== Step 3: Train Eagle3 drafter ==="
# --draft-arch llama       : drafter is a 1-layer Llama-shape attention block
#                            with verifier hidden_size / num_*_heads.
# --draft-hidden-act relu2 : Nemotron(-H) MLPs are squared-ReLU; matching the
#                            drafter activation gives faster convergence.
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" .venv/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --speculator-type eagle3 \
    --draft-arch llama \
    --draft-hidden-act relu2 \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing generate \
    --on-generate delete

echo "Drafter checkpoint at: $OUTPUT_DIR/checkpoints/"
```

For plain Nemotron (e.g. `nvidia/Nemotron-Nano-12B-v2`), use the same script
with `MODEL` updated, drop `--enforce-eager`, and keep `--draft-hidden-act relu2`.

## Step 3 — Serve the trained drafter

Speculators tags the output config with `architectures: ["Eagle3LlamaForCausalLM"]`
and `speculators_model_type: "eagle3"`, so vLLM's
[`SpeculatorsConfig`](../../../vllm/transformers_utils/configs/speculators/base.py)
handles it transparently:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --trust-remote-code \
    --enforce-eager \
    --speculative-config '{
        "model": "./output_nemotron3_nano_30b_a3b_eagle3/checkpoints",
        "num_speculative_tokens": 4,
        "method": "eagle3"
    }'
```

For DFlash, swap `--speculator-type eagle3` → `--speculator-type dflash` in
the training script and `"method": "eagle3"` → `"method": "dflash"` in the
serve command.

## Step 4 — Measure acceptance

```bash
.venv/bin/python tests/v1/e2e/spec_decode/measure_acceptance.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --draft ./output_nemotron3_nano_30b_a3b_eagle3/checkpoints \
    --num-prompts 80 --output-tokens 256
```

Treat anything below ~30% position-0 acceptance as a sign that one of the
following is wrong:

1. The aux-layer choice is suboptimal. Try setting an explicit triple via
   `speculators_config.eagle_aux_hidden_state_layer_ids`. For Nemotron-3-Nano-30B-A3B
   the candidate set is `(5, 12, 19, 26, 33, 42)`; experiment with
   `(19, 26, 33)` (later-bias) or `(5, 26, 42)` (wider spread).
2. `--draft-hidden-act` is wrong (must match `mlp_hidden_act` in the
   verifier's config — `relu2` for both Nemotron and Nemotron-H).
3. FP8 noise is bleeding into training. Re-train with `--noise-std 0.0`,
   or train against the BF16 release and serve the FP8 release.
4. Sample budget too low. 5k ShareGPT samples is a smoke test; bump to
   ≥200k samples or switch to UltraChat-200k.

## Mandatory pre-PR checks (per `AGENTS.md`)

```bash
gh issue list --repo vllm-project/vllm        --search "Nemotron Eagle3"        --limit 20
gh pr list    --repo vllm-project/vllm        --state open --search "nemotron_h SupportsEagle3"
gh pr list    --repo vllm-project/vllm        --state open --search "Nemotron-H Eagle3"
gh issue list --repo vllm-project/speculators --search "nemotron"
gh pr list    --repo vllm-project/speculators --state open --search "nemotron"
```

If any of those return an active PR covering the same surface, do **not**
open a duplicate; rebase onto / collaborate on the existing one.

The PR description for this work must include:

- Why this is not duplicating the existing PRs found above.
- The exact `pre-commit run` and `pytest` commands you ran, and their
  results.
- A clear statement that AI assistance was used (the commit already carries
  `Co-authored-by: Claude` and `Made-with: Cursor` trailers).
- Acceptance numbers from Step 4 on a non-trivial dataset (≥80 prompts).
  Without numbers the change is "infrastructure only" — that's fine, but
  state it explicitly so reviewers don't expect speedups.

## Known caveats / open work

- **MTP for `Nemotron-3-Nano-30B-A3B`**: the released checkpoint ships no
  `mtp.*` weights, so `NemotronHMTPModel` cannot be used out of the box.
  Speculators v0.5.0 has no MTP trainer; adding one is a separate (much
  larger) feature.
- **Mamba-2 state and Eagle3**: the drafter does not see the verifier's SSM
  state. We mitigate by tapping attention-layer outputs (which are the only
  layers whose hidden state is "stateless" w.r.t. SSM accumulation), but a
  drafter that consumes Mamba-2 hidden state directly may eventually beat
  this.
- **Nemotron-NAS**: a similar patch could be applied to
  `vllm/model_executor/models/nemotron_nas.py`. Not done in this PR; open
  a follow-up if/when needed.
- **Validation**: this PR is infrastructure only. The Eagle3 hooks are
  exercised by `tests/v1/spec_decode/test_speculators_eagle3.py` for Llama
  / Qwen3; we do not yet have a test fixture for Nemotron(-H). Add one
  in a follow-up PR alongside the first published Nemotron Eagle3 drafter.
