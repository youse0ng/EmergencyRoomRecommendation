from django.contrib import admin
from recommendations.models import Hospital,EmergencyReport

# Register your models here.
admin.site.register(Hospital)
admin.site.register(EmergencyReport)