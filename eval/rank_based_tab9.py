from pic_utils import *

'''
    rank-based uncertainty estimation (Table 9)
'''
x1, x2, x3, x4, x5 = get_test_record('record.csv')
reverse, count = 0, 0
for i in range(len(x1)):
    for j in range(i, len(x1)):
        if x1[i] < x1[j] and x2[i] > x2[j]:
            reverse += 1
        count += 1
print(reverse/count, count)

