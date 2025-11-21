from django.urls import path
from . import views

urlpatterns = [
    path('summarizer_index', views.index, name='summarizer_index'),
    path('pdf/', views.summarize_pdf, name='summarize_pdf'),
    path('docx/', views.summarize_docx, name='summarize_docx'),
    path('image/', views.summarize_image, name='summarize_image'),
    path('mp4/', views.summarize_mp4, name='summarize_mp4'),
    path('mp3/', lambda r: views.summarize_audio(r, "mp3"), name='summarize_mp3'),
    path('wav/', lambda r: views.summarize_audio(r, "wav"), name='summarize_wav'),
    path("audio/<str:fmt>/", views.summarize_audio, name="summarize_audio"),
    path('youtube/', views.summarize_youtube, name='summarize_youtube'),
]
