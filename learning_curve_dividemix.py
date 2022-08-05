import sys, re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

log_fname = sys.argv[1]
learning_curve_fname = log_fname.replace('.txt', '_lc.png')
print('producing %s...' % learning_curve_fname)

with open(log_fname, 'r') as fhand:
    acc_val_net1 = []
    acc_val_net2 = []
    for line in fhand:
        found = re.search('Validation.*Acc1:([0-9]*\.[0-9]*).*Acc2:([0-9]*\.[0-9]*)', line)
        if found:
            acc_val_net1.append(float(found.group(1)))
            acc_val_net2.append(float(found.group(2)))

print(len(acc_val_net1))
print(len(acc_val_net2))

fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(range(1, len(acc_val_net1)+1), acc_val_net1, label='net1 validation')
ax.plot(range(1, len(acc_val_net2)+1), acc_val_net2, label='net2 validation')
ax.set_xticks(np.arange(0, len(acc_val_net1)+1, 10))
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
ax.grid()
ax.legend(fontsize=16)
plt.suptitle('Learning Curve', fontsize=20)
fig.savefig(learning_curve_fname, bbox_inches='tight')
plt.close(fig)
