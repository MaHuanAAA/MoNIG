from pic_utils import *
import matplotlib.pyplot as plt

'''
    Select 200 samples to show the noise detection, Fig.4.
'''
x1, x2, x3, x4, x5 = get_test_record('record.csv')
a1, a2, b1, b2 = [], [], [], []
cnt1, cnt2 = 0, 0
for i in range(0, len(x1)):
    if x1[i] == 1:
        if cnt1 == 200:
            break
        a1.append(x1[i])
        b1.append(x2[i])
        cnt1 += 1
    else:
        if cnt2 == 200:
            break
        a2.append(x1[i])
        b2.append(x2[i])
        cnt2 += 1

a1, b1, a2, b2 = get_distinguish_points(a1, b1, a2, b2)

# drawing
plt.scatter(a1, b1, s=3**2, color='orange', label='noise on View 1')
plt.scatter(a2, b2, s=3**2, color='blue', label='noise on View 2')
plt.xlabel(u"Uncertainty of View 1", fontsize='18')
plt.ylabel(u"Uncertainty of View 2", fontsize='18')
plt.legend(loc='upper right', prop={'size': 16})
plt.plot([0, 1], [0, 1], color="black", linewidth=3, linestyle="--")
plt.xlim(-0.04, 1.04)
plt.ylim(-0.04, 1.04)
plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.18)
plt.tick_params(labelsize=18)
plt.show()
