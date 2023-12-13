
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from skintestApp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.homePage, name='home'),
    path('aboutus/', views.aboutus, name='about'),
    path('check/', views.check, name='check'),
    path('policy/', views.policy, name='policy'),
    path('terms/', views.terms, name='terms'),
    path('upload/', views.upload, name='upload'),
     path('agree/',views.agree,name='agree'),
    path('detect_skin_disease/', views.detect_skin_disease, name='detect_skin_disease'),  # Add this line
    path('skin_detection_view/', views.skin_detection_view, name='skin.detection_view'),
]

# Add the following lines at the end to serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
