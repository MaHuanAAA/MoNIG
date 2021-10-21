from scipy.interpolate import make_interp_spline
from sklearn import metrics
import numpy as np
import csv


def get_auroc_target(test_size):
    auroc_target = []
    for i in range(test_size):
        auroc_target.append(0)
    for i in range(3300):
        auroc_target.append(1)
    return auroc_target


def get_test_record(name):
    x, x2, x3, x4, x5 = [], [], [], [], []
    with open('../{}'.format(name), "rt", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i in reader:
            x.append(float(i[0]))
            x2.append(float(i[1]))
            x3.append(float(i[2]))
            x4.append(float(i[3]))
            x5.append(float(i[4]))
    return x, x2, x3, x4, x5


def get_roc(x):
    TPRs, FPRs = [], []
    ids = np.argsort(x)
    ids = ids[::-1]
    tp, fp = 0, 0
    for i in ids:
        if i<3300:
            fp += 1
        else:
            tp += 1
        TPR = tp/3300
        FPR = fp/3300
        TPRs.append(TPR)
        FPRs.append(FPR)
    return TPRs, FPRs


def get_roc_scores(x, x2, x3, x4, x5, target):
    s1 = metrics.roc_auc_score(target, x)
    s2 = metrics.roc_auc_score(target, x2)
    s3 = metrics.roc_auc_score(target, x3)
    s4 = metrics.roc_auc_score(target, x4)
    s5 = metrics.roc_auc_score(target, x5)
    return s1, s2, s3, s4, s5


def smooth(index, y):
    x_smooth = np.linspace(min(index), max(index), 300)
    y_smooth = make_interp_spline(index, y)(x_smooth)
    return x_smooth, y_smooth


def get_noise_uncertainty_points(au, eu, index, aumean, eumean, au2, eu2):
    aumax, eumax, aumin, eumin = [], [], [], []
    for i in range(len(index)):
        au[i] = au[i]/aumean[-1]
        eu[i] = eu[i] / eumean[-1]
        au2[i] = au2[i] / aumean[-1]
        eu2[i] = eu2[i] / eumean[-1]
        aumean[i] = aumean[i] / aumean[-1]
        eumean[i] = eumean[i] / eumean[-1]
        aumax.append(max(au[i], au2[i]))
        aumin.append(min(au[i], au2[i]))
        eumax.append(max(eu[i], eu2[i]))
        eumin.append(min(eu[i], eu2[i]))
    return aumax, eumax, aumin, eumin


def norm(x, maxx, minx):
    x_s = []
    for i in range(len(x)):
        x_s.append((x[i]-minx)/(maxx-minx))
    return x_s


def get_distinguish_points(a1, b1, a2, b2):
    maxa = max(a1)
    maxb = max(b2)
    mina1 = min(a1)
    minb1 = min(b1)
    a1 = norm(a1, maxa, mina1)
    b1 = norm(b1, maxb, minb1)
    a2 = norm(a2, maxa, mina1)
    b2 = norm(b2, maxb, minb1)
    return a1, b1, a2, b2

