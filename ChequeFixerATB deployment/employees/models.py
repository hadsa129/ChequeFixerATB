from django.db import models

class Employee(models.Model):
    employee_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    gender = models.CharField(max_length=10)
    user_type = models.CharField(max_length=50)
    sign_in_date = models.DateField()
    sign_in_time = models.TimeField()
    sign_out_time = models.TimeField()
    ip_address = models.GenericIPAddressField()
    connection_status = models.CharField(max_length=50)
    job_position = models.CharField(max_length=100)
    phone = models.CharField(max_length=20)
    email = models.EmailField()

    def __str__(self):
        return self.name
