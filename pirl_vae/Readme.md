Run: 

CUDA_VISIBLE_DEVICES=0 python3 train.py --ep 500 --bs 1024 --lr 3e-4 --steps 3 --z_dim 128 --wKLDMAX 0.0 --start_wKLD 0 --duration_wKLD 1 --cuda
