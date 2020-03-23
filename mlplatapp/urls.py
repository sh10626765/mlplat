from django.urls import path

from . import views

urlpatterns = [
    path('mlplat/', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('show/', views.show, name='show'),

    path('data/<str:name>', views.data, name='data'),

    path('statquality/<str:name>', views.data, name='statquality'),
    path('algoquality/<str:name>', views.data, name='algoquality'),

    path('download/<str:name>', views.download, name='download'),

    path('qualitycontrol/<str:data_name>', views.qualitycontrol, name='qualitycontrol'),
    path('edit/<str:name>/<int:number>', views.edit, name='edit'),

    path('test/', views.test, name='test'),
]
