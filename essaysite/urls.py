"""essaysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.urls import path
from django.contrib.auth import views
from user.views import user_page
from signup.views import signup
from essaygrader.views import FileFieldView, essay_preview, grade_essays
from gradebook.views import view_gradebook

urlpatterns = [
    path('', views.LoginView.as_view(), name='login'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('userpage', user_page, name='userpage'),
    path('signup', signup, name='signup'),
    path('upload', FileFieldView.as_view(), name='upload'),
    path('essaypreview', essay_preview, name='essaypreview'),
    path('gradeessays', grade_essays, name='gradeessays'),
    path('gradebook', view_gradebook, name='gradebook'),
    path('admin/', admin.site.urls),
]