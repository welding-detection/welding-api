from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import User, Inspection, InspectionDetail
from django.utils.decorators import method_decorator


TOKEN = 'justicehand1999'


# 회원가입
# from django.views.decorators.csrf import csrf_exempt
# @method_decorator(csrf_exempt, name='dispatch')
class SignupView(APIView):
    def post(self, request):
        login_id = request.data.get('login_id')
        password = request.data.get('password')
        name = request.data.get('name')

        user = User(login_id=login_id, password=password, name=name)
        user.save()

        print(request.data)

        if not (login_id and password and name):
            return Response({"success": False, "message": "Required fields are missing."},
                            status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({"success": True, "message": "User created successfully"}, status=status.HTTP_201_CREATED)


# 로그인
class LoginView(APIView):
    def post(self, request):
        login_id = request.data.get('login_id')
        password = request.data.get('password')
        user = User.objects.filter(login_id=login_id, password=password).first()

        if user:
            session_token = 'justicehand1999'
            user.session_token = session_token
            user.save()
            return Response({"loginStatus": True, "sessionToken": session_token}, status=status.HTTP_200_OK)
        else:
            return Response({"loginStatus": False, "message": "Invalid credentials"},
                            status=status.HTTP_401_UNAUTHORIZED)


# 이미지 업로드, 결함 탐지
class UploadView(APIView):
    def post(self, request):
        requester = User.objects.get(login_id=request.data.get('login_id'))
        inspection = Inspection(requester=requester)
        inspection.save()

        file_name = request.data.get('file_name')
        file_url = request.data.get('file_url')

        # --------------------------------
        file_result = "result..."  # 결함 탐지 로직 추가
        # --------------------------------

        detail = InspectionDetail(inspection=inspection, file_name=file_name, file_url=file_url,
                                  file_result=file_result)
        detail.save()

        return Response({"success": True, "message": "Defects detected successfully"}, status=status.HTTP_200_OK)


# 파일 목록 조회
class ListRetrievalView(APIView):
    def get(self, request, login_id):
        user = User.objects.get(login_id=login_id)
        inspections = Inspection.objects.filter(requester=user)

        data = [{"id": ins.id, "create_date": ins.create_date, "complete_flag": ins.complete_flag} for ins in
                inspections]

        return Response({"inspections": data}, status=status.HTTP_200_OK)


# 파일 상세 조회
class DetailRetrievalView(APIView):
    def get(self, request, inspection_id):
        details = InspectionDetail.objects.filter(inspection_id=inspection_id)

        data = [{"file_name": detail.file_name, "file_url": detail.file_url, "file_result": detail.file_result} for
                detail in details]

        return Response({"details": data}, status=status.HTTP_200_OK)
