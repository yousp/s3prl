CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python run_downstream.py \
  -m train \
  -u npc_local \
  -k result/pretrain/npc_film_ls100/states-epoch-56.ckpt \
  -d phone_linear \
  -n eval_npc_film_phone
