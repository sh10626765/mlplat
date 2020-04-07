import pandas as pd
import numpy as np
from mlplatapp import utils
import pymongo
import random

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
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

dfnd = df.iloc[:, 4:-1]
dfd = df.iloc[:, -1]

sample_num = len(df)
feature_num = len([i for i in dfnd])

non_decision_attr_mat = dfnd.values
decision_attr_mat = dfd.values

corrm = dfnd.corr(method='pearson')

for i in range(feature_num):
    for j in range(i + 1, feature_num):
        if corrm.iloc[i, j] > 0.8:
            print(i,j,corrm.iloc[i, j])
        # print(corrm.iloc[i, j])
# decision_attr_mat = dfd.values.reshape(-1, 1)
# print(dfnd.describe())
# c1 = 2
# c2 = 2
# x = [[]]
# v = [[]]
# for i in range(51):
#     for j in range(31):
#         x[i][j] = random.random()
#         v[i][j] = random.random()

# rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
# rfe.fit(non_decision_attr_mat, decision_attr_mat)
# print(rfe.ranking_)
# print(rfe.support_)
# clt = PCA(n_components=1)
# non = clt.fit_transform(non_decision_attr_mat)
#
# x = non.reshape(1, -1).tolist()[0]
# y = decision_attr_mat.reshape(1, -1).tolist()[0]
#
# clt = IsolationForest(n_estimators=500)
# labellist = clt.fit_predict(list(zip(x, y)))
# print(labellist)

#
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
