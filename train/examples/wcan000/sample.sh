CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES

python ../../../src/learn.py --seed 0\
       --n_class 10\
       --batch_size 256\
       --n_latent 10\
       --running_mode sampling\
       --from_example mnist\
       --load_index 19\
       --wgan\
       --can
