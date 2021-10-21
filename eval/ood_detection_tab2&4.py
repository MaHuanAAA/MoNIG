from pic_utils import *
import matplotlib.pyplot as plt

'''
    Draw auroc curves for OOD detection (Table 2 & 4).
    There are test_size in distribution samples and the same number of OOD samples,
    test records are save in record.csv during training. 
'''
test_size = 8375  # for CT slices dataset, test_size = 8375
target = get_auroc_target(test_size)
x, x2, x3, x4, x5 = get_test_record('record.csv')
t1, f1 = get_roc(x)
t2, f2 = get_roc(x2)
t3, f3 = get_roc(x3)
t4, f4 = get_roc(x4)
t5, f5 = get_roc(x5)
s1, s2, s3, s4, s5 = get_roc_scores(x, x2, x3, x4, x5, target)


# Drawing curves
plt.xlabel(u"FPR", fontsize='20')
plt.ylabel(u"TPR", fontsize='20')
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.18)
plt.tick_params(labelsize=18)
plt.plot(f1, t1, color="blue", label='Gaussian(area = %0.3f)' % s1, linewidth=3)
plt.plot(f2, t2, color="green", label='EVD-AU(area = %0.3f)' % s2, linewidth=3)
plt.plot(f4, t4, label='EVD-EU(area = %0.3f)' % s3, linewidth=3)
plt.plot(f3, t3, color="red", label='Ours-AU(area = %0.3f)' % s4, linewidth=3)
plt.plot(f5, t5, label='Ours-EU(area = %0.3f)' % s5, linewidth=3)
plt.plot([0, 1], [0, 1], color="yellow", linestyle="--", linewidth=3)
plt.legend(loc='lower right', prop={'size': 10})

plt.show()



