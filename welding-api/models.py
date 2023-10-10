from django.db import models
from django.contrib.auth.models import AbstractUser


# 장고 기본... 확장
# class CustomUser(AbstractUser):
#     session_token = models.CharField(max_length=256, blank=True, null=True)

class User(models.Model):
    login_id = models.CharField(max_length=255, unique=True, verbose_name="사용자 ID")
    password = models.TextField(verbose_name="사용자 비밀번호")
    name = models.CharField(max_length=255, verbose_name="사용자 이름")
    create_date = models.DateTimeField(auto_now_add=True, verbose_name="사용자 생성일자")


class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    arc_stick_disconnect_location = models.JSONField()
    detected_message = models.CharField(max_length=256)


class Inspection(models.Model):
    requester = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="요청자")
    create_date = models.DateTimeField(auto_now_add=True, verbose_name="검사일자")
    complete_flag = models.BooleanField(default=False, verbose_name="검사결과여부")


class InspectionDetail(models.Model):
    inspection = models.ForeignKey(Inspection, on_delete=models.CASCADE, verbose_name="검사테이블")
    file_name = models.CharField(max_length=255, verbose_name="파일명")
    file_url = models.TextField(verbose_name="파일 url")
    file_result = models.TextField(verbose_name="파일검사결과 url")
    result_comment = models.TextField(blank=True, null=True, verbose_name="명장 코멘트")
    analyze_comment = models.TextField(blank=True, null=True, verbose_name="분석 코멘트")

