from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from . import views
from . import views_train

urlpatterns = [
    path('', views.index),
    path("start-training/", views_train.start_training, name="start_training"),
    path("get-progress/", views_train.get_progress, name="get_progress"),
    path("get-results/", views_train.get_results, name="get_results"),
    path('dashboard/', views.dashboard, name='dashboard'),
    path("list_models/", views.list_models, name="list_models"),
    path('dashboard/run_training/', views.start_training, name='run_training'),
    path('dashboard/run_testing/', views.run_testing, name='run_testing'),
    path('dashboard/run_evaluation/', views.run_evaluation, name='run_evaluation'),
    path('admin/', admin.site.urls),
    path('classifier/', include('classifier.urls')),
    path('summarizer/', include('summarizer.urls')),
    path("clustering/", include("clustering.urls"))
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)