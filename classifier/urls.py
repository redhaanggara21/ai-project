from django.urls import path
from . import views
from . import views_kfold_evaluate
from . import views_train
from . import views_upload
from .views_training_classification import (
    start_training_classification, 
    get_training_progress, 
    get_training_results,
    run_testing,
    run_kfold_evaluation,
    list_models
)

urlpatterns = [
    path('', views.home_view, name='home'),
    path('train/', views.train_page, name='train_page'),
    path('upload-page/', views_upload.upload_page, name='upload_page'),
    path('upload-process/', views_upload.upload, name='upload-process'),
    path('start-training/', views_train.start_training, name='start_training'),
    path('get-results/', views_train.get_results, name='get_results'),
    path('get-progress/', views_train.get_progress, name='get_progress'),
    path('stop-training/', views_train.stop_training, name='stop_training'),
    path('start-evaluation/', views_kfold_evaluate.start_kfold_evaluate, name='start_evaluation'),
    path('get-progress-training/', views_train.get_progress, name='get_progress_training'),
    path('get-progress-upload/', views_upload.get_progress, name='get_progress_upload'),
    path('get-progress-evaluate/', views_upload.get_progress, name='get_progress_upload'),
    path('download/<str:model_type>/', views_train.download_model, name='download_model'),
    path('kfold/', views.kfold_page, name='kfold_page'),
    path('kfold/start/', views_kfold_evaluate.start_kfold_evaluate, name='start_kfold_evaluate'),
    path('kfold/progress/', views_kfold_evaluate.get_kfold_progress, name='get_kfold_progress'),
    
    # URL Classifier yang benar (tanpa awalan 'classifier/' karena sudah di-include di root)
    path('train-classification/', start_training_classification, name='start_training_classification'),
    path('get-progress-classification/', get_training_progress, name='get_training_progress'),
    path('get-results-classification/', get_training_results, name='get_training_results'),
    path('run-testing/', run_testing, name='run_testing'),
    path('run-kfold/', run_kfold_evaluation, name='run_kfold_evaluation'),
    path('list-models/', list_models, name='list_models'),
]