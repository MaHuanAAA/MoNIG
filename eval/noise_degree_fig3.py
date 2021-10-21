import matplotlib
from pic_utils import *
import matplotlib.pyplot as plt

'''
    Show the relationship between the uncertainty and noise degree Fig.3 (b)(c).
    Noise degree are shown as index.
'''
x1, x2, x3, x4, x5 = get_test_record('record.csv')
index = [0.1, 0.2, 0.3, 0.4, 0.5]
test_size = 8375
au, eu, au2, eu2, aumean, eumean = [], [], [], [], [], []
ausum, eusum, ausum2, eusum2 = 0, 0, 0, 0
for i in range(test_size, 2*test_size):
    ausum += x1[i]
    eusum += x2[i]
    ausum2 += x3[i]
    eusum2 += x4[i]
    if (i-test_size) % (test_size/len(index)) == 0:
        au.append(ausum/(test_size/len(index)))
        eu.append(eusum/(test_size/len(index)))
        aumean.append((ausum+ausum2) / (2*test_size/len(index)))
        eumean.append((eusum+eusum2) / (2*test_size/len(index)))
        au2.append(ausum2 / (test_size/len(index)))
        eu2.append(eusum2 / (test_size/len(index)))
        ausum, eusum, ausum2, eusum2 = 0, 0, 0, 0


aumax, eumax, aumin, eumin = get_noise_uncertainty_points(au, eu, index, aumean, eumean, au2, eu2)


# drawing
myfont = matplotlib.font_manager.FontProperties(size=12)
plt.plot(index, eumean, label='EU', linewidth=3)
plt.plot(index, aumean, label='AU', linewidth=3)
plt.fill_between(index, eumin, eumax, facecolor='#00BFFF', alpha=0.5, interpolate=True)
plt.fill_between(index, aumin, aumax, facecolor='#FFA500', alpha=0.5, interpolate=True)
plt.xlabel(u"Noise ε", fontsize='20')
plt.legend(loc='upper left', prop={'size': 18})
plt.ylabel(u"Uncertainty", fontsize='20')
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
plt.tick_params(labelsize=18)
plt.show()
