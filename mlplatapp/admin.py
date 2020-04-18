from django.contrib import admin

# Register your models here.
from .models import Data
from .models import MachineLearningMethods

admin.site.register(Data)
admin.site.register(MachineLearningMethods)

admin.site.site_header = '锂电池材料数据机器学习平台管理'
admin.site.site_title = '登录管理平台'
admin.site.index_title = '后台管理'
