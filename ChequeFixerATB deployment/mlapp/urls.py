from django.urls import path
 # Ensure these views are correctly imported
from django.conf import settings
# urls.py

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
  
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
