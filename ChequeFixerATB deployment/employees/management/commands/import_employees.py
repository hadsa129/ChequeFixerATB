import csv
from django.core.management.base import BaseCommand
from employees.models import Employee

class Command(BaseCommand):
    help = 'Import employees from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str)

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.stdout.write(f'Processing row: {row}')  # Debugging line
                employee, created = Employee.objects.get_or_create(
                    employee_id=row['employee_id'],
                    defaults={
                    'name': row['name'],
                    'gender': row['gender'],
                    'user_type': row['user_type'],
                    'sign_in_date': row['sign_in_date'],
                    'sign_in_time': row['sign_in_time'],
                    'sign_out_time': row['sign_out_time'],
                    'ip_address': row['ip_address'],
                    'connection_status': row['connection_status'],
                    'job_position': row['job_position'],
                    'phone': row['phone'],
                    'email': row['email'],
                }
            )
                if created:
                    self.stdout.write(self.style.SUCCESS(f'Employee {employee.employee_id} created successfully'))
                else:
                    self.stdout.write(self.style.WARNING(f'Employee {employee.employee_id} already exists'))
