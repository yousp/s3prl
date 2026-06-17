#!/bin/bash

# Configuration Paths - Modify these if your dataset locations differ
LIBRISPEECH_PATH="/home/yousp08/datasets/LibriSpeech/LibriSpeech"
SPEECH_COMMANDS_PATH="/home/yousp08/datasets/speech_commands"
IEMOCAP_PATH="/groups/public/benchmark/IEMOCAP"

# Model Checkpoints
PRETRAINED_CKPT="result/pretrain/npc_film_ls100/states-50000.ckpt"
BASELINE_CKPT="../npc_360hr.ckpt"

# Action Selection
echo "================================================================="
echo " S3PRL Downstream Run Script (ASR, PR, SID, KS, ER) on RTX 3090"
echo "================================================================="
echo "Please choose an option:"
echo "1) Train ASR (FiLM-NPC)"
echo "2) Train ASR Baseline (Original NPC)"
echo "3) Auto-Resume/Evaluate ASR"
echo "-----------------------------------------------------------------"
echo "4) Train Phone Recognition / PR (FiLM-NPC)"
echo "5) Train PR Baseline (Original NPC)"
echo "6) Auto-Resume/Evaluate PR"
echo "-----------------------------------------------------------------"
echo "7) Train Speaker ID / SID (FiLM-NPC)"
echo "8) Train SID Baseline (Original NPC)"
echo "9) Auto-Resume/Evaluate SID"
echo "-----------------------------------------------------------------"
echo "10) Train Keyword Spotting / KS (FiLM-NPC)"
echo "11) Train KS Baseline (Original NPC)"
echo "12) Auto-Resume/Evaluate KS"
echo "-----------------------------------------------------------------"
echo "13) Train Emotion / ER (FiLM-NPC)"
echo "14) Train ER Baseline (Original NPC)"
echo "15) Auto-Resume/Evaluate ER"
echo "================================================================="
read -p "Enter option [1-15]: " OPTION

case $OPTION in
  1)
    echo "Starting ASR training (FiLM-NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$PRETRAINED_CKPT" -d asr -n eval_npc_film_asr \
      -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"
    ;;
  2)
    echo "Starting ASR training Baseline (Original NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$BASELINE_CKPT" -d asr -n eval_baseline_npc_asr \
      -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"
    ;;
  3)
    echo "ASR Actions:"
    echo "1) Auto-resume training"
    echo "2) Evaluate"
    read -p "Choose action [1-2]: " ACT
    if [ "$ACT" == "1" ]; then
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m train -d asr -n eval_npc_film_asr --auto_resume
    else
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m evaluate -e result/downstream/eval_npc_film_asr
    fi
    ;;
  4)
    echo "Starting Phone Recognition training (FiLM-NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$PRETRAINED_CKPT" -d phone_linear -n eval_npc_film_phone \
      -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"
    ;;
  5)
    echo "Starting Phone Recognition training Baseline (Original NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$BASELINE_CKPT" -d phone_linear -n eval_baseline_npc_phone \
      -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"
    ;;
  6)
    echo "PR Actions:"
    echo "1) Auto-resume training"
    echo "2) Evaluate"
    read -p "Choose action [1-2]: " ACT
    if [ "$ACT" == "1" ]; then
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m train -d phone_linear -n eval_npc_film_phone --auto_resume
    else
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m evaluate -e result/downstream/eval_npc_film_phone
    fi
    ;;
  7)
    echo "Starting Speaker ID training (FiLM-NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$PRETRAINED_CKPT" -d speaker_linear_utter_libri -n eval_npc_film_sid \
      -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"
    ;;
  8)
    echo "Starting Speaker ID training Baseline (Original NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$BASELINE_CKPT" -d speaker_linear_utter_libri -n eval_baseline_npc_sid \
      -o config.downstream_expert.datarc.libri_root="$LIBRISPEECH_PATH"
    ;;
  9)
    echo "Speaker ID Actions:"
    echo "1) Auto-resume training"
    echo "2) Evaluate"
    read -p "Choose action [1-2]: " ACT
    if [ "$ACT" == "1" ]; then
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m train -d speaker_linear_utter_libri -n eval_npc_film_sid --auto_resume
    else
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m evaluate -e result/downstream/eval_npc_film_sid
    fi
    ;;
  10)
    echo "Starting Keyword Spotting training (FiLM-NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$PRETRAINED_CKPT" -d speech_commands -n eval_npc_film_ks \
      -o "config.downstream_expert.datarc.speech_commands_root='$SPEECH_COMMANDS_PATH',,config.downstream_expert.datarc.speech_commands_test_root='$SPEECH_COMMANDS_PATH/test'"
    ;;
  11)
    echo "Starting Keyword Spotting training Baseline (Original NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$BASELINE_CKPT" -d speech_commands -n eval_baseline_npc_ks \
      -o "config.downstream_expert.datarc.speech_commands_root='$SPEECH_COMMANDS_PATH',,config.downstream_expert.datarc.speech_commands_test_root='$SPEECH_COMMANDS_PATH/test'"
    ;;
  12)
    echo "KS Actions:"
    echo "1) Auto-resume training"
    echo "2) Evaluate"
    read -p "Choose action [1-2]: " ACT
    if [ "$ACT" == "1" ]; then
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m train -d speech_commands -n eval_npc_film_ks --auto_resume
    else
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m evaluate -e result/downstream/eval_npc_film_ks
    fi
    ;;
  13)
    echo "Starting Emotion training (FiLM-NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$PRETRAINED_CKPT" -d emotion -n eval_npc_film_emotion \
      -o config.downstream_expert.datarc.root="$IEMOCAP_PATH"
    ;;
  14)
    echo "Starting Emotion training Baseline (Original NPC)..."
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
      -m train -u npc_local -k "$BASELINE_CKPT" -d emotion -n eval_baseline_npc_emotion \
      -o config.downstream_expert.datarc.root="$IEMOCAP_PATH"
    ;;
  15)
    echo "Emotion Actions:"
    echo "1) Auto-resume training"
    echo "2) Evaluate"
    read -p "Choose action [1-2]: " ACT
    if [ "$ACT" == "1" ]; then
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m train -d emotion -n eval_npc_film_emotion --auto_resume
    else
      CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py -m evaluate -e result/downstream/eval_npc_film_emotion
    fi
    ;;
  *)
    echo "Invalid option. Exiting."
    exit 1
    ;;
esac
