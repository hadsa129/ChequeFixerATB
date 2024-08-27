from django.shortcuts import render
from cheques.models import Cheque
from employees.models import Employee
from bank.models import Bank
from django.db.models import Sum, Count
from django.shortcuts import render, redirect
from django.http import HttpResponse
from mlapp.forms import ChequeUploadForm
from mlapp.models import ChequeData
from mlapp.ml_autocorrection import preprocess_and_extract_ocr, correct_text_using_best_method
import os
from django.conf import settings
from datetime import datetime
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from accounts.forms import CustomUserCreationForm
from accounts.models import CustomUser

@login_required
def profile_view(request):
    user = request.user
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('profile')
    else:
        form = CustomUserCreationForm(instance=user)

    context = {
        'form': form,
    }
    return render(request, 'admin_dashboard/profile.html', context)

def convert_date_format(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None

def upload_image(request):
    if request.method == 'POST':
        form = ChequeUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            media_root = settings.MEDIA_ROOT
            upload_dir = os.path.join(media_root, 'uploads')

            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            image_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.name}"
            image_path = os.path.join(upload_dir, image_name)

            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            ocr_results = preprocess_and_extract_ocr(image_path)
            corrected_text, is_correct = correct_text_using_best_method(image_path)

            converted_date = convert_date_format(ocr_results.get('date', ''))

            existing_cheque = ChequeData.objects.filter(cheque_id=ocr_results['id']).first()
            if existing_cheque:
                return redirect('details', cheque_id=existing_cheque.cheque_id)
            else:
                cheque_data = ChequeData(
                    cheque_id=ocr_results['id'],
                    client_name=ocr_results.get('client', ''),
                    amount_digits=ocr_results.get('amount', ''),
                    amount_words=ocr_results.get('words', ''),
                    date=converted_date,
                    corrected_amount=corrected_text,
                    is_correct=is_correct,
                    image=image  # Save the image directly
                )
                cheque_data.save()

                return redirect('admin_dashboard/details.html', cheque_id=cheque_data.cheque_id)
    else:
        form = ChequeUploadForm()

    # Adjust the template path here:
    return render(request, 'admin_dashboard/upload.html', {'form': form})



from django.shortcuts import render, get_object_or_404

from django.shortcuts import render, get_object_or_404

def cheque_details(request, cheque_id):
    cheque = get_object_or_404(ChequeData, cheque_id=cheque_id)
    return render(request, 'admin_dashboard/details.html', {'cheque': cheque})

def inf_page(request, cheque_id):
    cheque = get_object_or_404(ChequeData, cheque_id=cheque_id)
    return render(request, 'admin_dashboard/inf.html', {'cheque': cheque})

import json


def dashboard_home(request):
    # Statistiques des chèques
    total_cheques = Cheque.objects.count()
    total_amount_cheques = Cheque.objects.aggregate(Sum('amount'))['amount__sum']
    cheques_by_month = list(Cheque.objects.values('date__month').annotate(count=Count('id')))
    
    # Statistiques des employés
    total_employees = Employee.objects.count()
    employees_by_type = list(Employee.objects.values('user_type').annotate(count=Count('id')))
    
    # Statistiques des banques
    total_clients = Bank.objects.count()
    total_balance = Bank.objects.aggregate(Sum('balance'))['balance__sum']
    balances_by_region = list(Bank.objects.values('region').annotate(total_balance=Sum('balance')))

    # Convert Decimal to float
    if total_amount_cheques is not None:
        total_amount_cheques = float(total_amount_cheques)
    if total_balance is not None:
        total_balance = float(total_balance)
    for entry in balances_by_region:
        entry['total_balance'] = float(entry['total_balance'])

    context = {
        'total_cheques': total_cheques,
        'total_amount_cheques': total_amount_cheques,
        'cheques_by_month': json.dumps(cheques_by_month),
        'total_employees': total_employees,
        'employees_by_type': json.dumps(employees_by_type),
        'total_clients': total_clients,
        'total_balance': total_balance,
        'balances_by_region': json.dumps(balances_by_region),
    }
    return render(request, 'admin_dashboard/dashboard_home.html', context)


def employee_management(request):
    return render(request, 'admin_dashboard/employee_management.html')



def bank_statistics(request):
    return render(request, 'admin_dashboard/bank_statistics.html')

def cheque_statistics(request):
    return render(request, 'admin_dashboard/cheque_statistics.html')

def employee_statistics(request):
    return render(request, 'admin_dashboard/employee_statistics.html')


def signout_view(request):
    # Usually, you'd perform the signout operation here or redirect to a signout URL
    return render(request, 'accounts/home.html')
cheques = Cheque.objects.all()
def cheque_table(request):
    
    return render(request, 'admin_dashboard/cheque_table.html', {'cheques': cheques})
employees = Employee.objects.all()
def employee_table(request):
    
    return render(request, 'admin_dashboard/employee_table.html', {'employees': employees})
banks = Bank.objects.all()
def bank_table(request):
    
    return render(request, 'admin_dashboard/bank_table.html', {'banks': banks})
