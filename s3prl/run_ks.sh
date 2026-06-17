#!/bin/bash
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k result/pretrain/npc_film_ls100/states-50000.ckpt \
  -d speech_commands -n eval_npc_film_ks \
  -o "config.downstream_expert.datarc.speech_commands_root='/home/yousp08/datasets/speech_commands',,config.downstream_expert.datarc.speech_commands_test_root='/home/yousp08/datasets/speech_commands/test'"
