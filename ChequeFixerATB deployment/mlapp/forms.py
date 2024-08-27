from django import forms

class ChequeUploadForm(forms.Form):
    image = forms.ImageField()
