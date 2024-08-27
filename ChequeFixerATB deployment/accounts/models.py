from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    EMPLOYEE = 'employee'
    ADMIN = 'admin'
    USER_TYPE_CHOICES = [
        (EMPLOYEE, 'Employee'),
        (ADMIN, 'Admin'),
    ]
    
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default=EMPLOYEE)
    employee_id = models.CharField(max_length=100, unique=True, blank=True, null=True)

    def __str__(self):
        return self.username

