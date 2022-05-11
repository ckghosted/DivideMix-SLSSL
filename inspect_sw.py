import sys
import numpy as np
import _pickle as cPickle

sw_fpath = sys.argv[1]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_path = '/data1/cclin/cifar-10-batches-py'
T_rw = 5.0

def unpickle(file):    
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

train_label=[]
for n in range(1,6):
    dpath = '%s/data_batch_%d'%(data_path,n)
    data_dic = unpickle(dpath)
    train_label = train_label+data_dic['labels']

sw = np.load(sw_fpath)
sw = 1/(1 + np.exp(-sw * T_rw))

for i in range(len(class_names)):
    idx_considered = [j for j in range(len(train_label)) if train_label[j] == i]
    sw_considered = sw[idx_considered]
    print('average sample weights for %s: %.2f' % (class_names[i], np.mean(sw_considered)))
