# Research efficient vits

Change inside train.py code the model you want to import
main.py is a smaller version of timm training code.

'''
torchrun --nproc_per_node=8 --nnodes=1 train.py ~/datasets/imagenet/ --aa rand-m9-mstd0.5-inc1 --cutmix 0.2 --color-jitter 0.0 --batch-size 256 --epochs 300 --warmup-epochs 5 --weight-decay 0.025 --workers 16 --lr 3e-3 --sched cosine --opt AdamW
'''