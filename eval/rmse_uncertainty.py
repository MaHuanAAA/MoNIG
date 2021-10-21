import matplotlib.pyplot as plt
from pic_utils import *


'''
    Relationship between the uncertainty and RMSE (Fig. 5  in supplementary).
'''
x1, x2, y, x4, x5 = get_test_record('record.csv')
scaled = 2387
index = np.argsort(y)
msemax= np.sqrt(y[index[4599]])
xl, xl2, yl, yl2 = [], [], [], []
ysum, ausum, eusum = 0, 0, 0
cnt = 0
for j in range(10):
    for i in range(len(index)):
        id = index[i]
        if 0.1*j*msemax<np.sqrt(y[id])<=0.1*(j+1)*msemax:
            ausum += x[id]
            eusum += x2[id]
            cnt += 1
    xl.append(0.1*(j+1)*msemax)
    yl.append(ausum/cnt)
    yl2.append(scaled*eusum/cnt)  # scaled eu and au


width = 0.3
for i in range(len(xl)):
    xl2.append(xl[i]+width)
plt.bar(xl2, yl2, width=width, label='EU')
plt.bar(xl, yl, width=width, label='AU')
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
plt.xlabel(u"RMSE", fontsize='20')
plt.ylabel(u"Uncertainty", fontsize='20')
plt.tick_params(labelsize=18)
plt.legend(loc='upper left')
plt.show()

