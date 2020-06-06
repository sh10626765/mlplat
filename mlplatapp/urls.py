from django.urls import path

from . import views

urlpatterns = [
    path('', views.upload, name='index'),
    path('upload/', views.upload, name='upload'),  # 上传逻辑
    path('show/', views.show, name='show'),  # 显示数据库中的数据集
    path('descinfo/<str:name>', views.info, name='descinfo'),  # 显示数据集的具体描述信息

    path('data/<str:name>', views.data, name='data'),  # 数据接口，用于直接获取数据库中的数据，主要给ajax用
    path('statquality/<str:name>', views.data, name='statquality'),
    path('algoquality/<str:name>', views.data, name='algoquality'),

    path('download/<str:name>', views.download, name='download'),  # 下载逻辑

    path('qualitycontrol/<str:data_name>', views.qualitycontrol, name='qualitycontrol'),  # 数据质量检测逻辑

    path('domainknowledgeembedding/<str:data_name>', views.chooseDomainKnowledgeEmbeddingMethod,
         name='domainknowledgeembedding'),  # 选择领域知识嵌入方式

    path('featureselection/<str:data_name>', views.featureselection, name='featureselection'),  # 特征核特征选择
    path('featureselectionbycontrirules/<str:data_name>', views.featureselectionBycontributionRules,
         name='featureselectionbycontrirules'),  #
    path('featureselectionbyscore/<str:data_name>', views.featureselectionByScore, name='featureselectionbyscore'),
    path('machinelearning/<str:data_name>/<str:from_where>', views.machinelearning, name='machinelearning'),
    # 展示机器学习模型参数
    path('predict/<str:data_name>/<str:method_name>/<str:from_where>', views.predict, name='predict'),  # 使用模型性能预测逻辑

    path('edit/<str:name>/<int:number>', views.edit, name='edit'),  # 编辑数据逻辑
    path('setverbose/<str:name>/<int:number>/<str:method>', views.setverbose, name='setverbose'),  # 标记高相关特征

    path('test/', views.test, name='test'),
]
