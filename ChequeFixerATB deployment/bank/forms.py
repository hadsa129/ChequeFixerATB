from django import forms
from .models import Bank

class BankForm(forms.ModelForm):
    class Meta:
        model = Bank
        fields = ['customer_id', 'name', 'surname', 'gender', 'age', 'region', 'job_classification', 'date_joined', 'balance']

