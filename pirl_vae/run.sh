#for wKLDMAX in 1e-6
#do
#  CUDA_VISIBLE_DEVICES=0 python3 train.py --ep 50 --bs 128 --lr 1e-4 --steps 1 --z_dim 64 --WD 1e-8 --wKLDMAX $wKLDMAX --start_wKLD 0 --duration_wKLD 50 --cuda --memo Final
#done

CUDA_VISIBLE_DEVICES=0 python3 train.py --ep 500 --bs 128 --lr 1e-4 --steps 10 --z_dim 64 --WD 1e-6 --wKLDMAX 5e-6 --start_wKLD 0 --duration_wKLD 50 --cuda --memo Final
