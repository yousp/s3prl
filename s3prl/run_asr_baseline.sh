#!/bin/bash
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k ../npc_360hr.ckpt \
  -d asr -n eval_baseline_npc_asr \
  -o config.downstream_expert.datarc.libri_root="/home/yousp08/datasets/LibriSpeech/LibriSpeech"
