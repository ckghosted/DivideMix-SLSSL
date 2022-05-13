import sys, re
import numpy as np

acc_fname = sys.argv[1]
acc_train_list = []
acc_test_list = []
acc_test_tch_list = []
with open(acc_fname, 'r') as fhand:
    for line in fhand:
        find_acc_train = re.search('\| Train Epoch #[0-9]*.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if find_acc_train:
            acc_train_list.append(float(find_acc_train.group(1)))
        find_acc_test = re.search('\| Test Epoch #[0-9]*.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if find_acc_test:
            acc_test_list.append(float(find_acc_test.group(1)))
        find_acc_test_tch = re.search('\| tch Test Epoch #[0-9]*.*Accuracy: ([0-9]*\.[0-9]*)\%', line)
        if find_acc_test_tch:
            acc_test_tch_list.append(float(find_acc_test_tch.group(1)))
print('Best Train: %.4f' % max(acc_train_list))
print('Last Train: %.4f' % np.mean(acc_train_list[-10:]))
print('Best Test: %.4f' % max(acc_test_list))
print('Last Test: %.4f' % np.mean(acc_test_list[-10:]))
if len(acc_test_tch_list) > 0:
    print('Best Test tch: %.4f' % max(acc_test_tch_list))
    print('Last Test tch: %.4f' % np.mean(acc_test_tch_list[-10:]))
