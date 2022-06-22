#!/user/bin/env/ bash
nohup  python -u  new_main.py \
 --data=Games \
 --gpu=3 \
 --epoch=20 \
 --hidden_size=50 \
 --batchSize=50 \
 --lr=0.001 \
 --l2=0.0001 \
 --layer_num=2 \
 --item_max_length=50 \
 --user_max_length=50 \
 --attn_drop=0.3 \
 --feat_drop=0.3 \
 --record \
 >./results/ga_results 2>./results/ga_error&
