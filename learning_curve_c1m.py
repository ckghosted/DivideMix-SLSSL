import sys, re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

log_fname = sys.argv[1]
skip_ep = int(sys.argv[2])
learning_curve_fname = log_fname.replace('_log.txt', '_lc.png')
print('producing %s...' % learning_curve_fname)

with open(log_fname, 'r') as fhand:
    acc_train_net1 = []
    acc_train_net2 = []
    acc_val_net1 = []
    acc_val_net2 = []
    for line in fhand:
        found = re.search('\| Train net1 Epoch.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if found:
            acc_train_net1.append(float(found.group(1)))
        found = re.search('\| Train net2 Epoch.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if found:
            acc_train_net2.append(float(found.group(1)))
        found = re.search('\| Validation.*Net1.*Acc: ([0-9]*\.[0-9]*)\%', line)
        if found:
            acc_val_net1.append(float(found.group(1)))
        found = re.search('\| Validation.*Net2.*Acc: ([0-9]*\.[0-9]*)\%', line)
        if found:
            acc_val_net2.append(float(found.group(1)))

acc_train_net1 = acc_train_net1[skip_ep:]
acc_train_net2 = acc_train_net2[skip_ep:]
print(len(acc_train_net1))
print(len(acc_train_net2))
print(len(acc_val_net1))
print(len(acc_val_net2))

fig, ax = plt.subplots(1, 1, figsize=(8,6))
if len(acc_train_net1) > 0:
    ax.plot(range(1, len(acc_train_net1)+1), acc_train_net1, label='net1 training')
    ax.set_xticks(np.arange(0, len(acc_train_net1)+1, 10))
if len(acc_train_net2) > 0:
    ax.plot(range(1, len(acc_train_net2)+1), acc_train_net2, label='net2 training')
    ax.set_xticks(np.arange(0, len(acc_train_net2)+1, 10))
if len(acc_val_net1) > 0:
    ax.plot(range(1, len(acc_val_net1)+1), acc_val_net1, label='net1 validation')
    ax.set_xticks(np.arange(0, len(acc_val_net1)+1, 10))
if len(acc_val_net2) > 0:
    ax.plot(range(1, len(acc_val_net2)+1), acc_val_net2, label='net2 validation')
    ax.set_xticks(np.arange(0, len(acc_val_net2)+1, 10))
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
ax.grid()
ax.legend(fontsize=16)
plt.suptitle('Learning Curve', fontsize=20)
fig.savefig(learning_curve_fname, bbox_inches='tight')
plt.close(fig)
