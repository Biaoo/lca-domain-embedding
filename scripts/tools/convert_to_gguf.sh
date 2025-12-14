#!/bin/bash
# convert_to_gguf.sh - helper script to export fine-tuned HF checkpoints to GGUF via llama.cpp

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$REPO_ROOT/../llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/data/output/lca-qwen3-st-finetuned}"
MODEL_ID="${MODEL_ID:-}"  # Optional HF repo id to download when MODEL_DIR is empty
GGUF_DIR="${GGUF_DIR:-$REPO_ROOT/data/output/gguf}"
OUT_BASENAME="${OUT_BASENAME:-$(basename "$MODEL_DIR")}"  # used for output names
OUTTYPE="${OUTTYPE:-f16}"  # target precision for the initial GGUF
QUANTS="${QUANTS:-Q8_0 Q6_K Q5_K_M Q4_K_M}"  # space separated list passed to llama-quantize
TEST_PROMPT="${TEST_PROMPT:-Say hello from GGUF conversion.}"

CONVERT_SCRIPT="$LLAMA_DIR/convert_hf_to_gguf.py"
LLAMA_QUANT="${LLAMA_DIR}/llama-quantize"
LLAMA_CLI="${LLAMA_DIR}/llama-cli"

if [[ ! -d "$LLAMA_DIR" ]]; then
  echo "[ERROR] llama.cpp checkout not found: $LLAMA_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
  echo "[ERROR] convert_hf_to_gguf.py missing at $CONVERT_SCRIPT" >&2
  echo "        Make sure you pulled llama.cpp next to lca-embedding." >&2
  exit 1
fi

mkdir -p "$GGUF_DIR"

if [[ -n "$MODEL_ID" && ! -d "$MODEL_DIR" ]]; then
  echo "Downloading $MODEL_ID with huggingface-cli ..."
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[ERROR] Model directory does not exist: $MODEL_DIR" >&2
  exit 1
fi

OUT_F16="$GGUF_DIR/${OUT_BASENAME}-${OUTTYPE}.gguf"

echo "============================================================"
echo "HF -> GGUF via llama.cpp"
echo "============================================================"
echo "Model dir : $MODEL_DIR"
echo "Output dir: $GGUF_DIR"
echo "Outtype   : $OUTTYPE"
echo "Quants    : $QUANTS"

echo "\n[1/3] Converting to ${OUTTYPE} GGUF ..."
python "$CONVERT_SCRIPT" "$MODEL_DIR" \
  --outtype "$OUTTYPE" \
  --outfile "$OUT_F16" \
  --sentence-transformers-dense-modules

if [[ ! -x "$LLAMA_QUANT" ]]; then
  echo "\n[WARN] $LLAMA_QUANT not found or not executable; skipping quantization." >&2
else
  echo "\n[2/3] Quantizing presets: $QUANTS"
  for quant in $QUANTS; do
    echo "  -> $quant"
    "$LLAMA_QUANT" "$OUT_F16" "$GGUF_DIR/${OUT_BASENAME}-${quant}.gguf" "$quant"
  done
fi

if [[ -x "$LLAMA_CLI" ]]; then
  echo "\n[3/3] Smoke test (Q4_K_M if available) ..."
  TEST_MODEL="$GGUF_DIR/${OUT_BASENAME}-Q4_K_M.gguf"
  if [[ -f "$TEST_MODEL" ]]; then
    "$LLAMA_CLI" -m "$TEST_MODEL" -p "$TEST_PROMPT" -n 64 || true
  else
    echo "  Skipping: $TEST_MODEL not found"
  fi
else
  echo "\n[INFO] llama-cli not present; skipping test run."
fi

echo "\nGGUF files in $GGUF_DIR:"
ls -lh "$GGUF_DIR"/*.gguf
