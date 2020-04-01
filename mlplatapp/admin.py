from django.contrib import admin

# Register your models here.
from .models import Data
from .models import MachineLearningMethods

admin.site.register(Data)
admin.site.register(MachineLearningMethods)
