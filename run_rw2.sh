for multi in 200
do
    for exp in -2 -1
    do
        for base in 1 2 5
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
                --lr_rw ${base}e${exp} \
                --fast_lr_rw 0.0001 \
                --diag_multi ${multi} \
                --num_rw 10 \
                > ./checkpoint/slssl3_cifar10_0.4_asym_u0_ep2w0_rw1_single_lrrw${base}e${exp}_d${multi}_log.txt
        done
    done
done
