#/usr/bin/bash
# filter option 1 : only target_threshold 
# filter option 2 : only topk 
# filter option 2 : target_threshold & topk

# OUT_MODEL_DIR='trained_models/hourglass/lossfilter1/'
# mkdir -p ${OUT_MODEL_DIR}
# log=$OUT_MODEL_DIR'log.txt'
# echo $log
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 0 --num_workers 12 \
#                     --target_threshold 7 --topk 20 --filter_option 1  | tee $log &

# OUT_MODEL_DIR='trained_models/hourglass/lossfilter2/'
# mkdir -p ${OUT_MODEL_DIR}
# log=$OUT_MODEL_DIR'log.txt'
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 1 --num_workers 12 \
#                     --target_threshold 8 --topk 20 --filter_option 1 | tee $log &

# OUT_MODEL_DIR='trained_models/hourglass/lossfilter3/'
# mkdir -p ${OUT_MODEL_DIR}
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 2 --num_workers 12 \
#                     --target_threshold 9 --topk 20 --filter_option 1 | tee $log &

# OUT_MODEL_DIR='trained_models/hourglass/lossfilter4/'
# mkdir -p ${OUT_MODEL_DIR}
# log=$OUT_MODEL_DIR'log.txt'
# echo $log
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 3 --num_workers 12 \
#                     --target_threshold 7 --topk 5 --filter_option 2  | tee "$log" &
                    
# OUT_MODEL_DIR='trained_models/hourglass/lossfilter5/'
# mkdir -p ${OUT_MODEL_DIR}  
# log=$OUT_MODEL_DIR'log.txt'
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 4 --num_workers 12 \
#                     --target_threshold 7 --topk 15 --filter_option 2  | > tee $log &

# OUT_MODEL_DIR='trained_models/hourglass/lossfilter6/'
# mkdir -p ${OUT_MODEL_DIR}
# log=$OUT_MODEL_DIR'log.txt'
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 5 --num_workers 12 \
#                     --target_threshold 7 --topk 10 --filter_option 3  | > tee $log &
                    
# OUT_MODEL_DIR='trained_models/hourglass/lossfilter7/'
# mkdir -p ${OUT_MODEL_DIR}    
# log=$OUT_MODEL_DIR'log.txt'
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 6 --num_workers 12 \
#                     --target_threshold 7 --topk 10 --filter_option 4  | > tee $log &
                    
# OUT_MODEL_DIR='trained_models/hourglass/lossfilter8/'
# mkdir -p ${OUT_MODEL_DIR}                               
# python train_paps.py --start_epoch 0 --end_epoch 120 --batch_size 24 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 7 --num_workers 12 \
#                     --target_threshold 8 --topk 5 --filter_option 4  | tee $log                    

e_epoch=120

# OUT_MODEL_DIR='trained_models/hourglass/lr1/'
# mkdir -p ${OUT_MODEL_DIR}
# log=$OUT_MODEL_DIR'log.txt'
# python train_paps.py --start_epoch 0 --end_epoch $e_epoch --batch_size 24 --learn_rate 0.001 \
#                     --saved_dir $OUT_MODEL_DIR --gpu_num 1 --num_workers 12 \
#                     --target_threshold 7 --topk 20 --filter_option 4  | tee $log &

OUT_MODEL_DIR='trained_models/hourglass/lr2/'
mkdir -p ${OUT_MODEL_DIR}
log=$OUT_MODEL_DIR'log.txt'
python train_paps.py --start_epoch 0 --end_epoch $e_epoch --batch_size 24 --learn_rate 0.0008 \
                    --saved_dir $OUT_MODEL_DIR --gpu_num 2 --num_workers 12 \
                    --target_threshold 7 --topk 20 --filter_option 4  | tee $log &
                    
OUT_MODEL_DIR='trained_models/hourglass/lr3/'
mkdir -p ${OUT_MODEL_DIR}
log=$OUT_MODEL_DIR'log.txt'
python train_paps.py --start_epoch 0 --end_epoch $e_epoch --batch_size 24 --learn_rate 0.0004 \
                    --saved_dir $OUT_MODEL_DIR --gpu_num 3 --num_workers 12 \
                    --target_threshold 7 --topk 20 --filter_option 4  | tee $log &    
                    
OUT_MODEL_DIR='trained_models/hourglass/lr3/'
mkdir -p ${OUT_MODEL_DIR}
log=$OUT_MODEL_DIR'log.txt'
python train_paps.py --start_epoch 0 --end_epoch $e_epoch --batch_size 24 --learn_rate 0.0001 \
                    --saved_dir $OUT_MODEL_DIR --gpu_num 4 --num_workers 12 \
                    --target_threshold 7 --topk 20 --filter_option 4  | tee $log &                          
