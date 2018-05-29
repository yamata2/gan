CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES

python ../../../src/learn.py --epoch 50\
       --learning_rate 0.00005\
       --save_interval 10\
       --sample_interval 1\
       --batch_size 128\
       --n_latent 10\
       --n_class 10\
       --seed 0\
       --running_mode training\
       --from_example mnist\
       --alternative_g_loss
