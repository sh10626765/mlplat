from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('show/', views.show, name='show'),

    path('data/<str:name>', views.data, name='data'),
    path('statquality/<str:name>', views.data, name='statquality'),
    path('algoquality/<str:name>', views.data, name='algoquality'),

    path('download/<str:name>', views.download, name='download'),

    path('qualitycontrol/<str:data_name>', views.qualitycontrol, name='qualitycontrol'),

    path('domainknowledgeembedding/<str:data_name>', views.chooseDomainKnowledgeEmbeddingMethod,
         name='domainknowledgeembedding'),

    path('featureselection/<str:data_name>', views.featureselection, name='featureselection'),
    path('featureselectionbycontrirules/<str:data_name>', views.featureselectionBycontributionRules,
         name='featureselectionbycontrirules'),
    path('featureselectionbyscore/<str:data_name>', views.featureselectionByScore, name='featureselectionbyscore'),
    path('machinelearning/<str:data_name>/<str:from_where>', views.machinelearning, name='machinelearning'),
    path('predict/<str:data_name>/<str:method_name>/<str:from_where>', views.predict, name='predict'),

    path('edit/<str:name>/<int:number>', views.edit, name='edit'),
    path('setverbose/<str:name>/<int:number>/<str:method>', views.setverbose, name='setverbose'),

    path('test/', views.test, name='test'),
]
