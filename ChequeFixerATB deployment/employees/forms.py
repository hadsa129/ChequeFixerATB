from django import forms
from .models import Employee

class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['employee_id', 'name', 'gender', 'user_type', 'sign_in_date', 'sign_in_time', 'sign_out_time', 'ip_address', 'connection_status', 'job_position', 'phone', 'email']
