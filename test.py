import pandas as pd
import numpy as np
from mlplatapp import utils
import pymongo
import random

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import preprocessing
import matplotlib.pyplot as plt


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def loaddata(name):
    if isinstance(name, str):
        return pd.read_excel(name)
    if isinstance(name, list):
        return pd.DataFrame(name)


filepath = 'E:\\毕设\\data\\traindata-0.5D0 - he.xlsx'

df = loaddata(filepath)

data_x = df.iloc[:, 4:-1]
data_y = df.iloc[:, -1]

sample_num = len(df)
feature_num = len([i for i in data_x])

corr = data_x.corr()

al = []

for i in range(feature_num):
    for j in range(i + 1, feature_num):
        if abs(corr.iloc[i, j]) > 0.8:
            al.append({i, j})
print(al)

al_d = {}

for i in range(feature_num):
    for j in al:
        if i in j:
            al_d[i] = al_d.get(i, 0) + 1

print(sorted(al_d.items(), key=lambda d: d[1], reverse=True))

for i in al:
    for j in al:
        if i & j:
            i.update(j)
aal = []
for i in al:
    if i not in aal:
        aal.append(i)
print(aal)


# x = dfnd.values
# y = dfd.values

# linea regression
# lr = LinearRegression()
# lr.fit(x, y)
# res = lr.predict(x)
#
# ms = sum([(i - j) ** 2 for i, j in zip(y, res.reshape(1, -1)[0])]) / sample_num
# print(ms)
# linea regression

# kfold
# sub = [i for i in range(feature_num) if np.random.randint(0, 2, feature_num)[i]]
# x = data_x.iloc[:, sub]
# print(x)
# kf = KFold(n_splits=5)
# loo=LeaveOneOut()
# mses = []
# maes = []
# index_of_sets = [[1, 3, 6, 14, 24, 28, 29, 38, 43, 48],
#                  [0, 5, 8, 15, 16, 25, 30, 31, 39, 44],
#                  [2, 10, 11, 17, 18, 26, 32, 33, 40, 45],
#                  [9, 12, 13, 19, 20, 27, 34, 35, 41, 46],
#                  [4, 7, 21, 22, 23, 36, 37, 42, 47, 49]]
# for train, test in loo.split(data_x):
#     print(train, test)
# for i in range(len(index_of_sets)):
#     lr = LinearRegression()
#     train_index = np.delete(range(50), index_of_sets[i])        # 获取训练集的样本编号
#     test_index = index_of_sets[i]                               # 获取验证集的样本编号
#     print(train_index, test_index)
#     lr.fit(data_x[train_index], data_y[train_index])
#     y_pred = lr.predict(data_x[test_index])
#
#     mses.append(mean_squared_error(data_y[test_index], y_pred))
#     maes.append(mean_absolute_error(data_y[test_index], y_pred))
#
# print(sum(mses) / len(mses))
# print(sum(maes) / len(maes))

# kfold

# pair_dist_n = pairwise_distances(non_decision_attr_mat)
# pair_dist = pairwise_distances(decision_attr_mat)
#
# dist_n_list = []
# dist_list = []
#
# n_min_idx = (0, 0)
# n_max_idx = (0, 0)
#
# n_max_val = 0
# n_min_val = np.inf
#
# for i in range(sample_num):
#     for j in range(i + 1, sample_num):
#         dist_n_list.append(pair_dist_n[i][j])
#         dist_list.append(pair_dist[i][j])
#         if pair_dist_n[i][j] < n_min_val:
#             n_min_val = pair_dist_n[i][j]
#             n_min_idx = (i, j)
#         if pair_dist_n[i][j] > n_max_val:
#             n_max_val = pair_dist_n[i][j]
#             n_max_idx = (i, j)
#
# dist_n_upper = np.quantile(dist_n_list, 0.75)
# dist_n_lower = np.quantile(dist_n_list, 0.25)
# dist_upper = np.quantile(dist_list, 0.9)
# dist_lower = np.quantile(dist_list, 0.1)
#
# print(dist_n_upper, dist_lower)
#
# temp = []
# tempt = {}
#
# for i in range(sample_num):
#     for j in range(i + 1, sample_num):
#         if pair_dist_n[i][j] >= dist_n_upper and pair_dist[i][j] <= dist_lower:
#             temp.append(i)
#             temp.append(j)
#             print(i, j)
#             print(pair_dist_n[i][j], pair_dist[i][j])
#             print(decision_attr_mat[i], decision_attr_mat[j])
#             print('########################################################################')
#
#         if pair_dist_n[i][j] <= dist_n_lower and pair_dist[i][j] >= dist_upper:
#             temp.append(i)
#             temp.append(j)
#             print(i, j)
#             print(pair_dist_n[i][j], pair_dist[i][j])
#             print(decision_attr_mat[i], decision_attr_mat[j])
#             print('########################################################################')
#
# for i in temp:
#     if temp.count(i) > 1:
#         tempt[i] = temp.count(i)
# for i in tempt:
#     print(i)
# tempt = sorted(tempt.items(), key=lambda x: x[1], reverse=True)
#
# print(tempt)
