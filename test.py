import pandas as pd
from mlplatapp import utils
import pymongo

from sklearn.neighbors import LocalOutlierFactor

filepath = 'E:\\毕设\\data\\traindata-0.5D0 - he.xlsx'

df=pd.read_excel(filepath)

dfnd=df.iloc[:,4:-1]
dfd=df.iloc[:,-1]

clf=LocalOutlierFactor(contamination=.1)
abc=clf.fit_predict(dfnd.values)

for i in range(len(abc)):
    if abc[i]==-1:
        print(i,dfd[i])