#!/bin/bash
../.venv/bin/python run_downstream.py \
  -m train -u npc_local -k ../npc_360hr.ckpt \
  -d speech_commands -n eval_baseline_npc_ks \
  -o "config.downstream_expert.datarc.speech_commands_root='/home/yousp08/datasets/speech_commands',,config.downstream_expert.datarc.speech_commands_test_root='/home/yousp08/datasets/speech_commands/test'"
