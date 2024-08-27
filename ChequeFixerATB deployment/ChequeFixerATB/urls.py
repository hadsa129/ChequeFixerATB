from django.contrib import admin
from django.urls import path, include
from accounts import views as accounts_views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),  # Inclure les URLs de l'application accounts
    path('', accounts_views.home, name='home'),    # Route pour la page d'accueil
    # Autres URL patterns
    path('cheques/', include('cheques.urls')),
     path('admin_dashboard/', include('admin_dashboard.urls')),
    path('employee_dashboard/', include('employee_dashboard.urls')),

    path('employees/', include('employees.urls')),
    path('mlapp/', include('mlapp.urls')),
    # Gestion des fichiers statiques
   
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)