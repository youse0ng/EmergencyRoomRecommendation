from django.shortcuts import render
from recommendations.models import EmergencyReport


def report(request):
    # EmergencyReport 데이터 가져오기
    reports = EmergencyReport.objects.all()
    
    # 템플릿에 전달할 context 생성
    context = {
        'reports': reports
    }
    
    # 템플릿 렌더링
    return render(request, 'recommendations/report.html', context)
