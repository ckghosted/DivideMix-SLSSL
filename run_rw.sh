for multi in 10 20 100 200
do
    for base in 1 2 5
    do
        for minus_exp in 3 4
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
                --n_rw_epoch 1 \
                --fast_lr_rw ${base}e-${minus_exp} \
                --diag_multi ${multi} \
                --num_rw 10 \
                > ./checkpoint/slssl3_cifar10_0.4_asym_u0_ep2w0_rw1_single_${base}e${minus_exp}_d${multi}_log.txt
        done
    done
done
