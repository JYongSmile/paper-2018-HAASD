# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 16:38
# @Author  : Barry
# @Email   : s.barry1994@foxmail.com
# @File    : noise_svm.py
# @Software: PyCharm Community Edition

from sklearn import svm
import time


from datetime import datetime
from swallowsound.swallowsound_input_data import read_data_sets
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score

import time
from datetime import datetime
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

dir = '/tmp/tensorflow/noise/input_2data250_fd'

# Import data
num_classes = 2
swallowsound = read_data_sets(dir,
                              gzip_compress=True,
                              train_imgaes='train-images-idx3-ubyte.gz',
                              train_labels='train-labels-idx1-ubyte.gz',
                              test_imgaes='t10k-images-idx3-ubyte.gz',
                              test_labels='t10k-labels-idx1-ubyte.gz',
                              one_hot=False,
                              validation_size=50,
                              num_classes=num_classes,
                              MSB=True)

batch_size = 20000
batch_x,batch_y = swallowsound.train.next_batch(batch_size)
test_x = swallowsound.test.images[:12000]
test_y = swallowsound.test.labels[:12000]

print (time.strftime('%Y-%m-%d %H:%M:%S') )
StartTime = time.clock()

for i in range(10,1000,10):
    # 传递训练模型的参数，这里用默认的参数
    clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03,max_iter=i)
    # clf = machine_learning.SVC(C=8.0, kernel='rbf', gamma=0.00,cache_size=8000,probability=False)
    # 进行模型训练
    clf.fit(batch_x, batch_y)
    # test
    # 测试集测试预测结果
    y_pred_rf = clf.predict(test_x)
    acc_rf = accuracy_score(test_y, y_pred_rf)
    print("%s n_estimators = %d, random forest accuracy:%f" % (datetime.now(), i, acc_rf))
    print('precision:', precision_score(test_y, y_pred_rf))
    print("recall:", recall_score(test_y, y_pred_rf))


EndTime = time.clock()
print('Total time %.2f s' % (EndTime - StartTime))


y_scores = cross_val_predict(clf, test_x, test_y, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(test_y, y_scores)

# plt.plot(thresholds, precisions, label='precisions',linewidth=2)
# plt.plot(thresholds, recalls, label='recalls', linewidth=2)
# plt.axis([0, thresholds, 0, 1])
# #plt.xlabel('False Positive Rate')
# plt.ylabel('thresholds')
# plt.show()


fpr, tpr, thresholds = roc_curve(test_y, y_scores)
print(roc_auc_score(test_y, y_scores))


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


plot_roc_curve(fpr, tpr)
plt.show()