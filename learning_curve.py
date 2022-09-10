import sys, re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

log_fname = sys.argv[1]
learning_curve_fname = log_fname.replace('_log.txt', '_lc.png')
print('producing %s...' % learning_curve_fname)

with open(log_fname, 'r') as fhand:
    loss_train = []
    loss_val = []
    loss_val_tch = []
    acc_train = []
    acc_val = []
    acc_val_tch = []
    acc_test = []
    acc_test_tch = []
    ep_counter = 1
    for line in fhand:
        find_train = re.search('Iter\[  1/[0-9]+\].*Loss: ([0-9]*\.[0-9]*).*Acc@1: ([0-9]*\.[0-9]*)\%', line)
        if find_train:
            loss_train.append(float(find_train.group(1)))
            acc_train.append(float(find_train.group(2)))
        find_val = re.search('\| Validation.*#  0.*Loss: ([0-9]*\.[0-9]*) Acc@1: ([0-9]*\.[0-9]*)\%', line)
        find_val_for_baseline = re.search('\| Validation Epoch #([0-9]*)\s*Loss: ([0-9]*\.[0-9]*) Acc@1: ([0-9]*\.[0-9]*)\%', line)
        if find_val:
            loss_val.append(float(find_val.group(1)))
            acc_val.append(float(find_val.group(2)))
        elif find_val_for_baseline:
            if ep_counter == int(find_val_for_baseline.group(1)):
                loss_val.append(float(find_val_for_baseline.group(2)))
                acc_val.append(float(find_val_for_baseline.group(3)))
                ep_counter = ep_counter + 1
        find_val_tch = re.search('\| tch Validation.*#  0.*Loss: ([0-9]*\.[0-9]*) Acc@1: ([0-9]*\.[0-9]*)\%', line)
        if find_val_tch:
            loss_val_tch.append(float(find_val_tch.group(1)))
            acc_val_tch.append(float(find_val_tch.group(2)))
        find_train = re.search('\| Train Epoch #[0-9]*.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if find_train:
            acc_train.append(float(find_train.group(1)))
        find_test = re.search('\| Test Epoch #[0-9]*.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if find_test:
            acc_test.append(float(find_test.group(1)))
        find_test_tch = re.search('\| tch Test Epoch #[0-9]*.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if find_test_tch:
            acc_test_tch.append(float(find_test_tch.group(1)))

# print(len(loss_train))
# print(len(loss_val))
# print(len(loss_val_tch))
if len(acc_train) >= 2 * len(acc_test):
    acc_train = [acc_train[i] for i in range(len(acc_train)) if i % 2 == 0]
print(len(acc_train))
print(len(acc_test))
print(len(acc_test_tch))

fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(range(1, len(acc_train)+1), acc_train, label='training')
ax.plot(range(1, len(acc_test)+1), acc_test, label='testing')
if len(acc_test_tch) > 0:
    ax.plot(range(1, len(acc_test_tch)+1), acc_test_tch, label='testing (tch)')
ax.set_xticks(np.arange(0, len(acc_test)+1, 50))
ax.set_xlabel('E-step Epochs', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
ax.grid()
ax.legend(fontsize=16)
plt.suptitle('Learning Curve', fontsize=20)
fig.savefig(learning_curve_fname, bbox_inches='tight')
plt.close(fig)
