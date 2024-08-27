from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='admin_dashboard_home'),
    path('employee_management/', views.employee_management, name='employee_management'),
    path('upload_cheques/', views.upload_image, name='upload_cheques'),
    path('bank_statistics/', views.bank_statistics, name='bank_statistics'),
    path('cheque_statistics/', views.cheque_statistics, name='cheque_statistics'),
    path('employee_statistics/', views.employee_statistics, name='employee_statistics'),
    path('bank_table/', views.bank_table, name='bank_table'),
    path('cheque_table/', views.cheque_table, name='cheque_table'),
    path('employee_table/', views.employee_table, name='employee_table'),
    path('details/<str:cheque_id>/', views.cheque_details, name='details'),
    path('inf/<str:cheque_id>/', views.inf_page, name='inf'),
    path('signout/', views.signout_view, name='signout'),
    path('profile/', views.profile_view, name='profile'),
]
