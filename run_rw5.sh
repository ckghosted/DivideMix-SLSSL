for num_rw in 0 2 5 10 20
do
    for multi in 1 10 100
    do
        for fast_lr_rw in 0.01 0.001 0.0001
        do
            python3 train.py \
                --lr 0.005 \
                --fast_lr 0.2 \
                --num_epochs 1 \
                --noise_mode asym \
                --r 0.4 \
                --lambda_u 0 \
                --kl_epoch 10 \
                --num_ssl 10 \
                --num_warmup 0 \
                --tch_model_1 slssl3_cifar10_0.4_asym_u0_ep200w100_tch1.pth.tar \
                --n_rw_epoch 3 \
                --lr_rw 0.2 \
                --fast_lr_rw ${fast_lr_rw} \
                --diag_multi ${multi} \
                --num_rw ${num_rw} \
                > ./checkpoint/slssl3_cifar10_0.4_asym_u0_ep1w0_rw1_single_fast${fast_lr_rw}_d${multi}_n${num_rw}log.txt
        done
    done
done
