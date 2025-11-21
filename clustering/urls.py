from django.urls import path
from . import views

urlpatterns = [
    path("cluster/", views.index, name="cluster_index"),
     path("upload/", views.cluster_image, name="cluster_image"),
]
