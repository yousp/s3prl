#!/bin/bash
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k result/pretrain/npc_film_ls100/states-50000.ckpt \
  -d asr -n eval_npc_film_asr \
  -o config.downstream_expert.datarc.libri_root="/home/yousp08/datasets/LibriSpeech/LibriSpeech"
