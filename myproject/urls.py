import os
from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve  # <-- Wajib diimpor untuk serve file lokal
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
    path("clustering/", include("clustering.urls")),
]

# Handler khusus untuk membuka file di folder 'models/' dan 'media/' saat DEBUG = True
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    
    # Menangkap URL dengan pola /models/<nama_file>
    urlpatterns += [
        re_path(r'^models/(?P<path>.*)$', serve, {
            'document_root': os.path.join(settings.BASE_DIR, 'models'),
        }),
    ]