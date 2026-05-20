from django.urls import path
from core.views import chat

urlpatterns = [
    path("", chat),
]