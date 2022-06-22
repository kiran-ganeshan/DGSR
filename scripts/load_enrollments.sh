#!/user/bin/env/ bash
nohup  python -u  new_data.py \
 --data=Enrollments \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --k_hop=3 \
 >./results/en_data 2>./results/en_data_error&