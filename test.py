import pandas as pd
import numpy as np
from mlplatapp import utils
import pymongo

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances


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


filepath = 'E:\\毕设\\data\\traindata-0.5D0 - he.xlsx'
df=pd.read_excel(filepath)
kk=[k.replace('.','').replace('$','') for k in df]
cl = pymongo.MongoClient('127.0.0.1', 27017)
db = cl.get_database(name='materialsData')
col = db.get_collection('traindata-0.5D0 - he3.xlsx')
attr=['111', 'Li7PSe6', 'shu.ssy_2019122611180312', 'Li7(PSe4)Se2', '1.98', '2.55', '10.5', '0.17', '2.19', '0.0198', '7.0', '10.475', '1149.44166', '9.1472', '5.4923', '2.633', '3.863', '2.63018', '2.653', '0.706316', '0.57132', '0.580692', '2.314', '2.164', '2.944', '0.66', '0.0', '0.66', '255.8193102', '0.0', '0.0', '0.0', '1.84', '2.55', '1.84', '2.55', '0.5078125']
newitem=dict(zip(kk,attr))
print(newitem)
item = col.find_one({'NO': '1'})
print(col.update_one({'NO': '1'}, {'$set': newitem}))
# a=col.find_one_and_update({'NO': attr[0]}, item)
# print(a)
# df = loaddata(filepath)
#
# dfnd = df.iloc[:, 4:-1]
# dfd = df.iloc[:, -1]
#
# sample_num = len(df)
#
# non_decision_attr_mat = dfnd.values
# decision_attr_mat = dfd.values.reshape(-1, 1)
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
