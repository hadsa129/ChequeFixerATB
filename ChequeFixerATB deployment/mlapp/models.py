# models.py

from django.db import models

class ChequeData(models.Model):
    cheque_id = models.CharField(max_length=100, unique=True)
    client_name = models.CharField(max_length=255)
    amount_digits = models.CharField(max_length=255)
    amount_words = models.TextField()
    date = models.DateField(null=True, blank=True)
    corrected_amount = models.CharField(max_length=255)
    is_correct = models.BooleanField(default=False)
    image = models.ImageField(upload_to='uploads/', null=True, blank=True)

    @property
    def image_path(self):
        return self.image.url if self.image else None

    def __str__(self):
        return self.cheque_id
