#!/user/bin/env/ bash
 nohup  python -u  new_main.py \
 --data=Enrollments \
 --gpu=3 \
 --epoch=20 \
 --batchSize=50 \
 --layer_num=3 \
 --lr=0.001 \
 --l2=0.00001 \
 --item_max_length=50 \
 --user_max_length=50 \
 --record \
 --model_record \
 >./results/en_results 2>./results/en_error&