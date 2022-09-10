for num_rw in 20
do
    for multi in 1 10 100
    do
        for lr_rw in 0.1 0.2 0.5
        do
            for fast_lr_rw in 0.01 0.001 0.0001
            do
                python3 train.py \
                    --lr 0.005 \
                    --fast_lr 0.1 \
                    --meta_lr 0.2 \
                    --num_epochs 1 \
                    --noise_mode asym \
                    --r 0.4 \
                    --lambda_u 0 \
                    --kl_epoch 10 \
                    --num_ssl 10 \
                    --num_warmup 0 \
                    --tch_model_1 fix_cifar10_0.4_asym_lr0.050000_u0_ep0w100_tch1.pth.tar \
                    --n_rw_epoch 1 \
                    --lr_rw ${lr_rw} \
                    --fast_lr_rw ${fast_lr_rw} \
                    --diag_multi ${multi} \
                    --num_rw ${num_rw} \
                    > ./checkpoint/fix_cifar10_0.4_asym_u0_w0ep1rw1_single_n${num_rw}_d${multi}_lr${lr_rw}fast${fast_lr_rw}_log.txt
            done
        done
    done
done
