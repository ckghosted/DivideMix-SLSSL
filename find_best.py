import os
import fnmatch
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fname_pattern = sys.argv[1]
if 'checkpoint' in fname_pattern:
    fname_pattern = os.path.basename(fname_pattern)

if len(sys.argv) > 2:
    print_threshold = float(sys.argv[2])
else:
    print_threshold = 0.0

for root, dirs, files in os.walk('./checkpoint'):
    for f in files:
        if fnmatch.fnmatch(f, fname_pattern):
            log_fname = os.path.join(root, f)
            with open(log_fname, 'r') as fhand:
                learning_curve_fname = log_fname.replace('_log.txt', '_auc.png')
                print('producing %s...' % learning_curve_fname)
                auc_list1 = []
                auc_list2 = []
                auc_bird_list1 = []
                auc_bird_list2 = []
                auc_cat_list1 = []
                auc_cat_list2 = []
                for line in fhand:
                    find_auc = re.search('\[reweighting\] AUC1:\s*([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_list1.append(auc)
                    find_auc = re.search('\[reweighting\] AUC2:\s*([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_list2.append(auc)
                    find_auc = re.search('\[reweighting\] AUC1 for bird:\s*([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_bird_list1.append(auc)
                    find_auc = re.search('\[reweighting\] AUC2 for bird:\s*([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_bird_list2.append(auc)
                    find_auc = re.search('\[reweighting\] AUC1 for cat:\s*([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_cat_list1.append(auc)
                    find_auc = re.search('\[reweighting\] AUC2 for cat:\s*([0-9]\.[0-9]*)', line)
                    if find_auc:
                        auc = float(find_auc.group(1))
                        auc_cat_list2.append(auc)
                if len(auc_list1) > 0 and auc_list1[-1] > print_threshold:
                    print('parsing {}'.format(f))
                    print('AUC={:.3f}'.format(auc_list1[-1]))
                    if len(auc_bird_list1) > 0:
                        print('AUC bird={:.3f}'.format(auc_bird_list1[-1]))
                    if len(auc_cat_list1) > 0:
                        print('AUC cat={:.3f}'.format(auc_cat_list1[-1]))
                if len(auc_list2) > 0 and auc_list2[-1] > print_threshold:
                    print('parsing {}'.format(f))
                    print('AUC={:.3f}'.format(auc_list2[-1]))
                    if len(auc_bird_list2) > 0:
                        print('AUC bird={:.3f}'.format(auc_bird_list2[-1]))
                    if len(auc_cat_list2) > 0:
                        print('AUC cat={:.3f}'.format(auc_cat_list2[-1]))
                print('len(auc_list1):', len(auc_list1))
                print('len(auc_list2):', len(auc_list2))
                print('len(auc_bird_list1):', len(auc_bird_list1))
                print('len(auc_bird_list2):', len(auc_bird_list2))
                print('len(auc_cat_list1):', len(auc_cat_list1))
                print('len(auc_cat_list2):', len(auc_cat_list2))
                fig, ax = plt.subplots(1, 1, figsize=(8,6))
                if len(auc_list1) > 0:
                    ax.plot(range(1, len(auc_list1)+1), auc_list1, label='Net1 (all)')
                if len(auc_list2) > 0:
                    ax.plot(range(1, len(auc_list2)+1), auc_list2, label='Net2 (all)')
                if len(auc_bird_list1) > 0:
                    ax.plot(range(1, len(auc_bird_list1)+1), auc_bird_list1, label='Net1 (bird)')
                if len(auc_bird_list2) > 0:
                    ax.plot(range(1, len(auc_bird_list2)+1), auc_bird_list2, label='Net2 (bird)')
                if len(auc_cat_list1) > 0:
                    ax.plot(range(1, len(auc_cat_list1)+1), auc_cat_list1, label='Net1 (cat)')
                if len(auc_cat_list2) > 0:
                    ax.plot(range(1, len(auc_cat_list2)+1), auc_cat_list2, label='Net2 (cat)')
                ax.set_xticks(np.arange(1, len(auc_list1)+1, 1))
                #ax.set_xticklabels(['0-1', '0-2', '0-3', '50-1', '50-2', '50-3', '100-1', '100-2', '100-3', '150-1', '150-2', '150-3', '200-1', '200-2', '200-3', '250-1', '250-2', '250-3'])
                ax.set_xlabel('Accumulated M-step Epochs', fontsize=16)
                ax.set_ylabel('AUC', fontsize=16)
                ax.grid()
                ax.legend(fontsize=16)
                plt.suptitle('Learning Curve', fontsize=20)
                fig.savefig(learning_curve_fname, bbox_inches='tight')
                plt.close(fig)
