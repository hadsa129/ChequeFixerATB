from django.db import models

class Cheque(models.Model):
    cheque_id = models.CharField(max_length=255, unique=True)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    date = models.DateField()
    words = models.TextField()
    client = models.TextField()

    def __str__(self):
        return f"{self.client} - {self.cheque_id}"