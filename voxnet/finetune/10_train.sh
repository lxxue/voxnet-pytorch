CUDA_VISIBLE_DEVICES=3 \
    python train.py \
    --training_fname shapenet10_test.tar \
    --testing_fname shapenet10_train.tar \
    --model feature \
    --log_dir log_10 \
    --num_classes 10 \
    --max_epoch 32 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --momentum 0.9 \
    --optimizer adam \
    --decay_step 4 \
    --decay_rate 0.8 \
    --saved_fname weight10 \
    --cont \
    --ckpt_dir ../baseline/log_30/ \
    --ckpt_fname weight30
