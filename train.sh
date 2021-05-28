#/usr/bin/bash

python train_paps.py --start_epoch 0 --end_epoch 100 --batch_size 24 \
                    --saved_dir 'trained_models/resnet101_320/' --gpu_num 0 --num_workers 12