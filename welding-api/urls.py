"""
URL configuration for welding_api project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.views.generic import RedirectView

from .views import SignupView, LoginView, UploadView, ListRetrievalView, DetailRetrievalView
from django.urls import path

# 기존 기환짱꺼
# urlpatterns = [
#     path("adin/", admin.site.urlsm),
# ]


urlpatterns = [
    path('', RedirectView.as_view(url='/login/')),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('upload/', UploadView.as_view(), name='upload'),
    path('list/<str:login_id>/', ListRetrievalView.as_view(), name='listRetrieval'),
    path('detail/<int:inspection_id>/', DetailRetrievalView.as_view(), name='detailRetrieval'),
]

