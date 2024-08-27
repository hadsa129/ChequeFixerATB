from django.db import models

from django.db import models

class Bank(models.Model):
    customer_id = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    region = models.CharField(max_length=255)
    job_classification = models.CharField(max_length=255)
    date_joined = models.DateField()
    balance = models.DecimalField(max_digits=12, decimal_places=2)

    def __str__(self):
        return f"{self.name} {self.surname}"
