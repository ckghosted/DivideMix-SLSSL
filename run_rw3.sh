for multi in 10 20
do
    for lr_rw in 0.2
    do
        for fast_lr_rw in 0.01 0.005 0.001 0.0005 0.0001 0.00005
        do
            python3 Train_cifar_slssl3.py \
                --lr 0.005 \
                --fast_lr 0.2 \
                --num_epochs 2 \
                --noise_mode asym \
                --r 0.4 \
                --lambda_u 0 \
                --kl_epoch 10 \
                --num_ssl 10 \
                --num_warmup 0 \
                --tch_model_1 slssl2_cifar10_0.4_asym_u0_ep100_tch1.pth.tar \
                --n_rw_epoch 10 \
                --lr_rw ${lr_rw} \
                --fast_lr_rw ${fast_lr_rw} \
                --diag_multi ${multi} \
                --num_rw 8 \
                > ./checkpoint/slssl3_cifar10_0.4_asym_u0_ep2w0_rw1_single_lrrw${lr_rw}fast${fast_lr_rw}_d${multi}_log.txt
        done
    done
done
