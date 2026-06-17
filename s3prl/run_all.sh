#!/bin/bash
set -e

# Configuration Paths - Modify these if your dataset locations differ
LIBRISPEECH_PATH="/home/yousp08/datasets/LibriSpeech/LibriSpeech"
SPEECH_COMMANDS_PATH="/home/yousp08/datasets/speech_commands"

# Model Checkpoints
PRETRAINED_CKPT="result/pretrain/npc_film_ls100/states-epochs-56.ckpt"
BASELINE_CKPT="../npc_360hr.ckpt"

# -------------------------------------------------------------
# 1. ASR (Automatic Speech Recognition)
# -------------------------------------------------------------
echo "=================================================="
echo "[ASR] 1/4 Training FiLM-NPC..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k "$PRETRAINED_CKPT" -d asr -n eval_npc_film_asr \
  -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"

echo "=================================================="
echo "[ASR] 2/4 Evaluating FiLM-NPC..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m evaluate -t "test-clean" -e result/downstream/eval_npc_film_asr

echo "=================================================="
echo "[ASR] 3/4 Training Baseline (Original NPC)..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k "$BASELINE_CKPT" -d asr -n eval_baseline_npc_asr \
  -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"

echo "=================================================="
echo "[ASR] 4/4 Evaluating Baseline..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m evaluate -t "test-clean" -e result/downstream/eval_baseline_npc_asr


# -------------------------------------------------------------
# 2. Keyword Spotting (Speech Commands)
# -------------------------------------------------------------
echo "=================================================="
echo "[KS] 1/4 Training FiLM-NPC..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k "$PRETRAINED_CKPT" -d speech_commands -n eval_npc_film_ks \
  -o "config.downstream_expert.datarc.speech_commands_root='$SPEECH_COMMANDS_PATH',,config.downstream_expert.datarc.speech_commands_test_root='$SPEECH_COMMANDS_PATH/test'"

echo "=================================================="
echo "[KS] 2/4 Evaluating FiLM-NPC..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m evaluate -t "test" -e result/downstream/eval_npc_film_ks

echo "=================================================="
echo "[KS] 3/4 Training Baseline (Original NPC)..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k "$BASELINE_CKPT" -d speech_commands -n eval_baseline_npc_ks \
  -o "config.downstream_expert.datarc.speech_commands_root='$SPEECH_COMMANDS_PATH',,config.downstream_expert.datarc.speech_commands_test_root='$SPEECH_COMMANDS_PATH/test'"

echo "=================================================="
echo "[KS] 4/4 Evaluating Baseline..."
echo "=================================================="
../.venv/bin/python run_downstream.py \
  -m evaluate -t "test" -e result/downstream/eval_baseline_npc_ks

echo "=================================================="
echo "All ASR and Keyword Spotting experiments completed!"
echo "=================================================="
