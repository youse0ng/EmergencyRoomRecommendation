from django.db import models

# Create your models here.
from django.db import models

class Hospital(models.Model):
    name = models.CharField(max_length=255)
    latitude = models.FloatField()
    longitude = models.FloatField()
    rating = models.FloatField()
    address = models.CharField(max_length=255, blank=True, null=True)
    
    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Hospital"
        verbose_name_plural = "Hospitals"

class EmergencyReport(models.Model):
    report_summary = models.TextField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    report_time = models.DateTimeField(auto_now_add=True)
    
    # 추천된 응급실 정보를 외래 키로 연결
    recommended_hospitals = models.ManyToManyField(Hospital, related_name='emergency_reports')

    def __str__(self):
        return f"Report at {self.report_time}"
    
