from django.conf.urls import url, include
from django.contrib import admin
from rest_framework import routers
from FakeNewsServer.IsFake import views
from django.conf.urls import url

router = routers.DefaultRouter()

urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^admin/', admin.site.urls),
    url(r'^is_fake/$', views.IsFake.as_view())
]
