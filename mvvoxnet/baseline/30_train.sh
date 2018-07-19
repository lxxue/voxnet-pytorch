CUDA_VISIBLE_DEVICES=1 \
    python train.py \
    --training_fname shapenet30_train.tar \
    --testing_fname shapenet30_test.tar \
    --model mvvoxnet \
    --log_dir log_30 \
    --num_classes 30 \
    --max_epoch 32 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --momentum 0.9 \
    --optimizer adam \
    --decay_step 4 \
    --decay_rate 0.8 \
    --saved_fname weight30 \
    #--cont \
    #--ckpt_dir \
    #--ckpt_fname 
