import csv
from django.core.management.base import BaseCommand
from datetime import datetime
from cheques.models import Cheque

class Command(BaseCommand):
    help = 'Import cheques from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str)

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    amount = row['amount']
                    date_str = row['date']
                    words = row['words']
                    client = row['client']
                    cheque_id = row['id']
                    
                    # Parse date
                    date = datetime.strptime(date_str, '%d/%m/%Y').date()
                    
                except KeyError as e:
                    self.stdout.write(self.style.ERROR(f'Missing field in CSV: {e}'))
                    continue
                except ValueError as e:
                    self.stdout.write(self.style.ERROR(f'Error processing row {row}: {e}'))
                    continue

                # Ensure correct field name is used in get_or_create
                cheque, created = Cheque.objects.get_or_create(
                    cheque_id=cheque_id,
                    defaults={
                        'amount': amount,
                        'date': date,
                        'words': words,
                        'client': client,
                    }
                )

                if created:
                    self.stdout.write(self.style.SUCCESS(f'Cheque {cheque.cheque_id} created successfully'))
                else:
                    self.stdout.write(self.style.WARNING(f'Cheque {cheque.cheque_id} already exists'))

        self.stdout.write(self.style.SUCCESS('Cheque import completed successfully'))
