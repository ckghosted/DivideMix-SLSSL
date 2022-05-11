import os
import fnmatch
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fname_pattern = sys.argv[1]
print_threshold = float(sys.argv[2])

for root, dirs, files in os.walk('./checkpoint'):
    for f in files:
        if fnmatch.fnmatch(f, fname_pattern):
            log_fname = os.path.join(root, f)
            with open(log_fname, 'r') as fhand:
                learning_curve_fname = log_fname.replace('_log.txt', '_auc.png')
                print('producing %s...' % learning_curve_fname)
                auc_list = []
                auc_bird_list = []
                auc_cat_list = []
                for line in fhand:
                    find_auc = re.search('\[reweighting\] AUC:([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_list.append(auc)
                    find_auc = re.search('\[reweighting\] AUC for bird:([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_bird_list.append(auc)
                    find_auc = re.search('\[reweighting\] AUC for cat:([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_cat_list.append(auc)
                if auc_list[-1] > print_threshold:
                    print('parsing {}'.format(f))
                    print('AUC={:.3f}'.format(auc_list[-1]))
                    if len(auc_bird_list) > 0:
                        print('AUC bird={:.3f}'.format(auc_bird_list[-1]))
                    if len(auc_cat_list) > 0:
                        print('AUC cat={:.3f}'.format(auc_cat_list[-1]))
                print(len(auc_list))
                print(len(auc_bird_list))
                print(len(auc_cat_list))
                fig, ax = plt.subplots(1, 1, figsize=(8,6))
                ax.plot(range(1, len(auc_list)+1), auc_list, label='AUC (all)')
                ax.plot(range(1, len(auc_bird_list)+1), auc_bird_list, label='AUC (bird)')
                ax.plot(range(1, len(auc_cat_list)+1), auc_cat_list, label='AUC (cat)')
                ax.set_xticks(np.arange(0, len(auc_list)+1, 1))
                ax.set_xlabel('Epochs', fontsize=16)
                ax.set_ylabel('AUC', fontsize=16)
                ax.grid()
                ax.legend(fontsize=16)
                plt.suptitle('Learning Curve', fontsize=20)
                fig.savefig(learning_curve_fname, bbox_inches='tight')
                plt.close(fig)
