import csv
from django.core.management.base import BaseCommand
from bank.models import Bank
from django.db import transaction
from datetime import datetime
from decimal import Decimal

class Command(BaseCommand):
    help = 'Import banks from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str)

    @transaction.atomic
    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    customer_id = row['Customer ID']
                    name = row['Name']
                    surname = row['Surname']
                    gender = row['Gender']
                    age = int(row['Age'])
                    region = row['Region']
                    job_classification = row['Job Classification']
                    
                    # Convert 'Date Joined' from 'DD.Mon.YY' to 'YYYY-MM-DD'
                    date_str = row['Date Joined']
                    date_joined = datetime.strptime(date_str, '%d.%b.%y').date()

                    # Convert 'Balance' to Decimal
                    balance = Decimal(row['Balance'])
                    
                except KeyError as e:
                    self.stdout.write(self.style.ERROR(f'Missing field in CSV: {e}'))
                    continue
                except ValueError as e:
                    self.stdout.write(self.style.ERROR(f'Error processing row {row}: {e}'))
                    continue

                bank, created = Bank.objects.get_or_create(
                    customer_id=customer_id,
                    defaults={
                        'name': name,
                        'surname': surname,
                        'gender': gender,
                        'age': age,
                        'region': region,
                        'job_classification': job_classification,
                        'date_joined': date_joined,
                        'balance': balance,
                    }
                )

                if created:
                    self.stdout.write(self.style.SUCCESS(f'Bank {bank.customer_id} created successfully'))
                else:
                    self.stdout.write(self.style.WARNING(f'Bank {bank.customer_id} already exists'))

        self.stdout.write(self.style.SUCCESS('Bank import completed successfully'))
