for num_rw in 0 10 20 30
do
    for multi in 100
    do
        for multi_off in 1
        do
            for lr_rw in 0.2
            do
                for fast_lr_rw in 0.001
                do
                    FILE=./checkpoint/fix_cifar10_0.4_asym_u0_pen0.0_w0ep1rw20_single_n${num_rw}d${multi}and${multi_off}_lr${lr_rw}fast${fast_lr_rw}_log.txt
                    if [ -f "$FILE" ]; then
                        touch $FILE
                    else
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
                            --tch_model_1 $1 \
                            --n_rw_epoch 20 \
                            --lr_rw ${lr_rw} \
                            --fast_lr_rw ${fast_lr_rw} \
                            --diag_multi ${multi} \
                            --offd_multi ${multi_off} \
                            --num_rw ${num_rw} \
                            --r_penalty 0.0 \
                            > $FILE
                    fi
                done
            done
        done
    done
done
